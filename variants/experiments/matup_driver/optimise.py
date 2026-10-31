"""MatUp's optimisation (main_optim.main_optim + train.train_model) as an eager
loop with one hook, the extra loss of experiment 4.

The letters are the changes against the submodule, listed in README.md.

The extra_loss hook (f):
    extra_factory(dataset, args, device)   called once, after the dataset (LR init
                                           + upsampled renders) is loaded
        -> extra_loss(model, dataset, step) -> (tensor, dict)
    The tensor is added to MatUp's loss every step; the dict's values land in
    the epoch record (only the last step's). extra_factory=None reproduces MatUp.
    The one producer is exp4_grf_stats/grf_stats.factory.
"""
import time

import numpy as np
import torch

from variants.experiments.matup_driver.upstream import (FullyConnected, LIGHT_RADIUS, MatUpDataset,
                                                        Renderer, light_utils, mtrain, render_utils,
                                                        testing)


def optimise(args, device, extra_factory=None, seed=1996, num_workers=8, log=print):
    """Returns (maps_eval, model, dataset, history). maps_eval is MatUp's own
    list [albedo_lin, metallic, normal_internal, alpha, position] over the
    valid (size - tile_width + 1)^2 region."""
    np.random.seed(seed)                       # main_optim.py:37-38, same seed
    torch.manual_seed(seed)
    Renderer.init_renderer(args, device)       # (k) device is passed in: no set_device, no prints

    model = FullyConnected(num_in=7, num_out=7, tile_width=args.tile_width,
                           nb_hidden_layers=args.hid_layer_size,
                           hidden_layer_size=args.hid_layer_width,
                           non_lin=args.non_linearity).float().to(device)
    # (b) capturable=False: main_optim.py:73 sets True, which only matters under graph capture
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_mse = torch.nn.MSELoss().to(device)

    # (d) the renders under args.render_dir / upsamplings_dir come from sr_cache
    dataset = MatUpDataset(args, device)
    # (h) MatUp would run on with whatever it found
    if dataset.GT_renders is None:
        raise RuntimeError("no upsampled renders in %s" % args.upsamplings_dir)
    if dataset.GT_renders.shape[-1] != args.nb_lights:
        raise RuntimeError("%d renders but nb_lights=%d (README note 1)"
                           % (dataset.GT_renders.shape[-1], args.nb_lights))
    # (c) drop_last=True and a worker count (main_optim.py:81-84 has neither)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         num_workers=num_workers,
                                         persistent_workers=num_workers > 0,
                                         shuffle=True, pin_memory=True, drop_last=True)
    wi_tab = (LIGHT_RADIUS * light_utils.generate_hemisphere_pts(
        "Fibonacci", sample_count=args.nb_lights, near_z=False)).float().to(device)
    # (f) the hook
    extra_loss = extra_factory(dataset, args, device) if extra_factory else None

    history = []
    step = 0
    t0 = time.perf_counter()
    # (a) eager: train.py:59-139 captures this step in two CUDA graphs and
    # replays them on two streams; the ops per step are the same five
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
            # (g) per-step float(): synchronises, MatUp accumulates on the GPU
            acc["matup"] += float(loss.detach())
            if extra_loss is not None:         # (f)
                e, extra_log = extra_loss(model, dataset, step)
                loss = loss + e
                acc["extra"] += float(e.detach())
            loss.backward()
            optimizer.step()
            step += 1
        # (g) a history record instead of train.py:147's print; (k) no
        # model.train() per epoch, no save_interval snapshot (train.py:84, 149-151)
        n = len(loader)
        rec = {"epoch": epoch, "loss_matup": acc["matup"] / n, "loss_extra": acc["extra"] / n,
               "seconds": time.perf_counter() - t0}
        rec.update({k: float(v) for k, v in extra_log.items()})
        history.append(rec)
        log("  epoch %2d  matup %.6f  extra %.6f  (%.0fs)" % (
            epoch, rec["loss_matup"], rec["loss_extra"], rec["seconds"]))

    maps_eval = testing.eval_model(args, model, dataset, device)   # train.py:153
    return maps_eval, model, dataset, history
