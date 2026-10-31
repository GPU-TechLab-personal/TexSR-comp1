"""Experiment 1's precise indicator, read off a finished MatUp run (an addition,
not a change to MatUp)."""
import numpy as np
import torch

from variants.experiments.matup_driver.upstream import LIGHT_RADIUS, light_utils, render_utils


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
