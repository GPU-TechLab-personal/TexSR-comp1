#!/usr/bin/env python3
"""
실험 5 (학습): training throughput on this GPU, measured the same way per method.

    python variants/experiments/speed_memory/bench_train.py
    python variants/experiments/speed_memory/bench_train.py --steps 30   # gate

  esrgan:<config>   one full optimisation step of experiments 2/3 (G + D + VGG,
                    and the render loss where configured) at the config's batch,
                    on synthetic tensors - measures the model, not the data loader
  matup             MatUp's per-material optimisation step (MLP + 100-light render
                    + l2,1), batch 128, using a real eval12 material's cache
                    (needs one experiment-1/4 run to exist; skipped otherwise)

Warm-up 10 steps, then `--steps` timed, CUDA-synchronised. Reports it/s, peak
memory, and for MatUp the implied seconds per material (10 epochs x
246016/128 steps). GPU-hours to a TARGET metric are not measured here - they
come from the real training logs: gpu_hours.py.
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.experiments.common import paths                          # noqa: E402
from variants.experiments.esrgan_map_degradation.arms import separate             # noqa: E402
from variants.experiments.esrgan_map_degradation.core import assemble             # noqa: E402
from variants.experiments.esrgan_map_degradation.core.config import load_config   # noqa: E402
from variants.experiments.esrgan_map_degradation.core.model import (MAPS, VGGPerceptual,   # noqa: E402
                                                         VGGStyleDiscriminator)
from variants.experiments.esrgan_render_loss.arms import render_loss     # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = sorted(glob.glob(os.path.join(_HERE, os.pardir, "esrgan_map_degradation", "configs", "*.yaml"))
                 + glob.glob(os.path.join(_HERE, os.pardir, "esrgan_render_loss", "configs", "*.yaml")))


def bench_esrgan(cfg_path, steps, dev):
    """The arm's G/D/criterion as the trainer builds them (core/assemble.py),
    except - as before - no pretrained init, D at its default width and G
    with the config's num_block only: this measures speed, not quality."""
    cfg = load_config(cfg_path, [])
    mc, tr = cfg["model"], cfg["train"]
    G = assemble.generator(mc["arch"], mc["input"], mc.get("num_block", 23)).to(dev)
    per_map = cfg["loss"]["per_map"]
    gan_maps = [m for m in MAPS if per_map.get(m, {}).get("gan", 0) > 0]
    discs = assemble.discriminators(cfg["disc"]["mode"], gan_maps, separate.owns_discriminators(mc),
                                    lambda c: VGGStyleDiscriminator(c)).to(dev)
    need_vgg = any(v.get("vgg", 0) > 0 for v in per_map.values())
    vgg = VGGPerceptual().to(dev) if need_vgg else None
    render = None
    if render_loss.applies(cfg["loss"]):
        render = render_loss.RenderLoss(cfg["loss"]["render"], dev, cfg["data"]["hr_patch"])
    crit = assemble.criterion(cfg["loss"], vgg, render)
    og = torch.optim.Adam(G.parameters(), 1e-4)
    od = torch.optim.Adam(discs.parameters(), 1e-4) if discs.active() else None
    B, p = tr["batch"], cfg["data"]["hr_patch"]
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[tr.get("amp", "bf16")]
    lr = {m: torch.rand(B, 3, p // 4, p // 4, device=dev) for m in MAPS}
    hr = {m: torch.rand(B, 3, p, p, device=dev) for m in MAPS}

    def step():
        og.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            sr = G(lr)
            loss, _ = crit.generator(sr, hr, discs, psnr_stage=False)
        loss.backward()
        og.step()
        if od is not None:
            od.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                ld = discs.loss(sr, hr)
            ld.backward()
            od.step()

    return run(step, steps), {"batch": B, "hr_patch": p}


def bench_matup(steps, dev):
    from variants.experiments import matup_driver as MD
    from variants.experiments.common.sr_cache import lr_render_dir, sr_dir
    cands = sorted(glob.glob(os.path.join(paths.SR_CACHE, "eval12", "*", "radiance_swinir_L100")))
    if not cands:
        return None, {"skipped": "no eval12 swinir cache yet (run exp1 run_C.py --set eval12 once)"}
    root = os.path.dirname(cands[0])
    name = os.path.basename(root)
    args = MD.matup_args(os.path.join(paths.MATERIALS, "eval12", name, "LR"), lr_render_dir(root, 100),
                         sr_dir(root, "swinir", 100), "/tmp/unused")
    MD.Renderer.init_renderer(args, dev)
    model = MD.FullyConnected(7, 7, 17, 4, 128, "LeakyReLU").float().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-4)
    ds = MD.MatUpDataset(args, dev)
    loader = iter(torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=8,
                                              pin_memory=True, drop_last=True))
    wi = (1.4 * MD.light_utils.generate_hemisphere_pts("Fibonacci", 100, near_z=False)).float().to(dev)
    mse = torch.nn.MSELoss()
    batches = [[t.to(dev) for t in next(loader)] for _ in range(8)]   # data loading excluded
    k = [0]

    def step():
        sv, gt, pos = batches[k[0] % 8]
        k[0] += 1
        opt.zero_grad(set_to_none=True)
        maps, data, mid = MD.mtrain.mlp_forward_pass(sv, model, args, pos, gt)
        r = MD.render_utils.batch_render_tn_explicit(*maps, wi)
        MD.mtrain.loss_fn(r, data, args, mse, maps, mid).backward()
        opt.step()

    r = run(step, steps)
    per_material = 10 * (ds.nb_tiles // 128) / r["it_per_s"]
    return r, {"material": name, "batch": 128, "seconds_per_material_10ep_eager": per_material}


def run(step, steps):
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"it_per_s": steps / dt, "steps": steps, "peak_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--no-matup", action="store_true")
    ap.add_argument("--out", default=os.path.join(paths.RUNS, "exp5"))
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    res = {"gpu": torch.cuda.get_device_name(0), "entries": {}}
    for c in args.configs:
        name = "esrgan:" + os.path.basename(c).replace(".yaml", "")
        r, info = bench_esrgan(c, args.steps, dev)
        r.update(info)
        res["entries"][name] = r
        print("  %-34s %6.2f it/s  peak %6.0f MB  batch %d" % (name, r["it_per_s"], r["peak_mem_mb"], r["batch"]),
              flush=True)
        torch.cuda.empty_cache()
    if not args.no_matup:
        r, info = bench_matup(args.steps, dev)
        if r is None:
            print("  matup skipped: " + info["skipped"])
        else:
            r.update(info)
            res["entries"]["matup"] = r
            print("  %-34s %6.2f it/s  peak %6.0f MB  -> %.0f s / material (10 epochs, eager)" % (
                "matup", r["it_per_s"], r["peak_mem_mb"], info["seconds_per_material_10ep_eager"]))
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "train_%s.json" % res["gpu"].replace(" ", "_")), "w"), indent=2)


if __name__ == "__main__":
    main()
