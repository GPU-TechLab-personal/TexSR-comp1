"""
MatUp's optimisation, driven from outside with an optional extra loss term.

What is reused verbatim from external/matup_egsr (imported, never edited):
    model.FullyConnected, dataLoader.MatUpDataset, train.mlp_forward_pass,
    train.loss_fn (l2,1 + metallic reg), render_utils.batch_render_tn_explicit,
    testing.eval_model, data_utils.output_maps_from_list
What differs from main_optim.py / train.train_model, and why:
    * eager loop instead of CUDA-graph capture. The graph freezes the loss
      expression, so an added term (experiment 4) cannot be captured without
      rewriting it. Eager executes the same ops; the numbers differ only by
      kernel scheduling and RNG consumption. The gate
      (experiments/gates/gate_matup_driver.py) measures that difference against
      a real `main.py` run instead of asserting it is zero.
    * Adam(capturable=False): capturable only matters under graph capture.
    * the LR/SR renders come from sr_cache (same files MatUp would write),
      and nothing is rmtree'd afterwards (main_optim.py:103-104 deletes them).

extra_factory(dataset, args, device) is called once the dataset (LR init +
upsampled renders) is loaded and returns extra_loss(model, dataset, step) ->
(tensor, dict), added to MatUp's loss every step. None reproduces MatUp.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from . import paths

paths.add_matup()
import matup.train as mtrain                      # noqa: E402
from matup import data_utils, light_utils, render_utils, testing   # noqa: E402
from matup.dataLoader import MatUpDataset         # noqa: E402
from matup.model import FullyConnected            # noqa: E402
from matup.renderer import Renderer               # noqa: E402

from .render import LIGHT_RADIUS                  # noqa: E402


def matup_args(lr_dir, render_dir, up_dir, out_dir, **over):
    """main.py's argparse defaults, with --ogl_normals (MatSynth is OpenGL)."""
    a = dict(upscaler="swinir", input_mat=lr_dir.rstrip("/") + "/",
             render_dir=render_dir.rstrip("/") + "/", upsamplings_dir=up_dir.rstrip("/") + "/",
             output_folder=out_dir.rstrip("/") + "/", output_init=False,
             tile_width=17, batch_size=128, nb_epochs=10, learning_rate=1e-4, alpha_reg=1e-1,
             save_interval=999, hid_layer_width=128, hid_layer_size=4,
             non_linearity="LeakyReLU", crop_size=128, AA_upscale=2, nb_lights=100,
             disp_val=0.01, light_falloff=False, opengl_normals=True, temperature=0.5,
             material_size=128, up_batch_size=10)
    a.update(over)
    return argparse.Namespace(**a)


def optimise(args, device, extra_factory=None, seed=1996, num_workers=8, log=print):
    """Returns (maps_eval, model, dataset, history). maps_eval is MatUp's own
    list [albedo_lin, metallic, normal_internal, alpha, position] over the
    valid (size - tile_width + 1)^2 region."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    Renderer.init_renderer(args, device)

    model = FullyConnected(num_in=7, num_out=7, tile_width=args.tile_width,
                           nb_hidden_layers=args.hid_layer_size,
                           hidden_layer_size=args.hid_layer_width,
                           non_lin=args.non_linearity).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_mse = torch.nn.MSELoss().to(device)

    dataset = MatUpDataset(args, device)
    if dataset.GT_renders is None:
        raise RuntimeError("no upsampled renders in %s" % args.upsamplings_dir)
    if dataset.GT_renders.shape[-1] != args.nb_lights:
        raise RuntimeError("%d renders but nb_lights=%d (README note 1)"
                           % (dataset.GT_renders.shape[-1], args.nb_lights))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         num_workers=num_workers,
                                         persistent_workers=num_workers > 0,
                                         shuffle=True, pin_memory=True, drop_last=True)
    wi_tab = (LIGHT_RADIUS * light_utils.generate_hemisphere_pts(
        "Fibonacci", sample_count=args.nb_lights, near_z=False)).float().to(device)
    extra_loss = extra_factory(dataset, args, device) if extra_factory else None

    history = []
    step = 0
    t0 = time.perf_counter()
    for epoch in range(1, args.nb_epochs + 1):
        acc = {"matup": 0.0, "extra": 0.0}
        extra_log = {}
        for svbrdf, gt, pos in loader:
            svbrdf = svbrdf.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            maps, data, mid = mtrain.mlp_forward_pass(svbrdf, model, args, pos, gt)
            render = render_utils.batch_render_tn_explicit(*maps, wi_tab)
            loss = mtrain.loss_fn(render, data, args, loss_mse, maps, mid)
            acc["matup"] += float(loss.detach())
            if extra_loss is not None:
                e, extra_log = extra_loss(model, dataset, step)
                loss = loss + e
                acc["extra"] += float(e.detach())
            loss.backward()
            optimizer.step()
            step += 1
        n = len(loader)
        rec = {"epoch": epoch, "loss_matup": acc["matup"] / n, "loss_extra": acc["extra"] / n,
               "seconds": time.perf_counter() - t0}
        rec.update({k: float(v) for k, v in extra_log.items()})
        history.append(rec)
        log("  epoch %2d  matup %.6f  extra %.6f  (%.0fs)" % (
            epoch, rec["loss_matup"], rec["loss_extra"], rec["seconds"]))

    maps_eval = testing.eval_model(args, model, dataset, device)
    return maps_eval, model, dataset, history


def init_maps(args, device):
    """MatUp's starting point (bilinear x4 of the LR maps) over the same valid
    region as its output - the 'did the optimisation help at all' reference."""
    Renderer.init_renderer(args, device)
    dataset = MatUpDataset(args, device)
    b = (args.tile_width - 1) // 2
    v = dataset.get_svbrdf_init()[b:-b, b:-b].to(device)
    pos = dataset.get_position3D()[b:-b, b:-b].to(device)
    maps = data_utils.get_renorm_clamped_maps(data_utils.maps_from_data(v, pos))
    return maps, dataset


def write_outputs(maps_eval, args):
    """MatUp's own 512^2 PNG outputs (8px zero border, DirectX normal)."""
    os.makedirs(args.output_folder, exist_ok=True)
    data_utils.output_maps_from_list([m.clone() for m in maps_eval], args, "_opt", "",
                                     pad_output=True)


def to_canonical(maps_eval, lr_height=None, full_size=512):
    """MatUp's internal maps -> canonical material (protocol.py), valid region only.

    MatUp does not output height. If `lr_height` (LR canonical height) is given,
    the height slot is filled with its bilinear x4 over the same region - what
    MatUp itself renders with (dataLoader.py:115-128) - and must be reported as
    such, never as a MatUp prediction."""
    import cv2
    albedo, metallic, normal, alpha = [m.detach().float().cpu() for m in maps_eval[:4]]
    out = {
        "basecolor": render_utils.gammaCorrection(albedo.clamp(0, 1)).numpy(),
        "metallic": metallic[..., 0].clamp(0, 1).numpy(),
        "roughness": torch.sqrt(alpha[..., 0].clamp(0, 1)).numpy(),
    }
    n = data_utils.torch_norm(normal, dimToNorm=-1).numpy().copy()
    n[..., 1] = -n[..., 1]                      # internal -> OpenGL
    out["normal"] = n
    if lr_height is not None:
        size = out["basecolor"].shape[0]
        b = (full_size - size) // 2
        h = cv2.resize(lr_height, (full_size, full_size), interpolation=cv2.INTER_LINEAR)
        out["height"] = h[b:full_size - b, b:full_size - b]
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}


@torch.no_grad()
def per_light_residuals(maps_eval, dataset, args, device, rel_k=(2.0, 3.0),
                        abs_tau=(0.02, 0.05, 0.1), chunk=10):
    """Experiment 1 (정밀): after optimisation, the residual of every light at
    every texel, r_w(x) = || R_w(m_hat)(x) - S(R_w)(x) ||_2 over RGB, in the
    display space both are stored in.

    'Discarded' has no single definition under l2,1 (every light keeps a unit-
    norm gradient); reported are the fraction of (texel, light) pairs whose
    residual exceeds k x the texel's median over lights, and absolute
    thresholds. Both, so the choice can be made later rather than baked in."""
    b = args.tile_width // 2
    wi = (LIGHT_RADIUS * light_utils.generate_hemisphere_pts(
        "Fibonacci", sample_count=args.nb_lights, near_z=False)).float().to(device)
    maps = [m.to(device) for m in maps_eval]
    sr = dataset.GT_renders[b:-b, b:-b].to(device)                 # (H, W, 3, L)
    res = []
    for i in range(0, wi.shape[0], chunk):
        r = render_utils.batch_render_tn(maps, wi[i:i + chunk])    # (H, W, 3, l)
        if r.ndim == 3:
            r = r[..., None]
        res.append(torch.linalg.norm(r - sr[..., i:i + chunk], dim=2))
    r = torch.cat(res, dim=-1)                                     # (H, W, L)
    med = r.median(dim=-1, keepdim=True).values.clamp_min(1e-6)
    out = {
        "per_light_mean": r.mean(dim=(0, 1)).cpu().tolist(),
        "mean": float(r.mean()), "median": float(r.median()),
        "p90": float(torch.quantile(r.flatten()[::7], 0.9)),
        "discard_rel": {("gt_%gx_median" % k): float((r > k * med).float().mean()) for k in rel_k},
        "discard_abs": {("gt_%g" % t): float((r > t).float().mean()) for t in abs_tau},
    }
    edges = np.logspace(-4, 0, 41)
    hist = torch.histc(torch.log10(r.clamp_min(1e-4)), bins=40, min=-4, max=0)
    out["hist_log10"] = {"edges": edges.tolist(), "counts": hist.cpu().tolist()}
    residual_map = r.mean(dim=-1).cpu().numpy().astype(np.float16)
    return out, residual_map


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
