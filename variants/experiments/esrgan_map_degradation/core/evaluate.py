#!/usr/bin/env python3
"""
Full-protocol evaluation of a trained multi-map ESRGAN (experiments 2, 3).

    python variants/experiments/esrgan_map_degradation/core/evaluate.py --run exp2/2-0_base_scratch
    python variants/experiments/esrgan_map_degradation/core/evaluate.py --run exp3/3-1_render --tile 32

Per eval12 material: every map, every protocol metric (metrics.evaluate:
per-map primary + render-space FLIP/LPIPS over 16 Fibonacci lights), SR maps
written as PNG. The claimed maps for ESRGAN are all five.

--tile N runs inference on non-overlapping N x N LR tiles and stitches them;
the seam band (--band px around every HR tile boundary) is then reported
separately from the interior (공통 프로토콜: tiling을 쓰면 경계 영역 오차를 따로 측정).

The generator is rebuilt from the checkpoint's own config (core/assemble.py),
so checkpoints of the old experiments/exp2_esrgan/train.py load unchanged.

Output: <run>/eval/<ckpt>[_tile<N>]/{results.json, summary.md, <material>/*.png}
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4)))
from variants.experiments.common import metrics as M, paths                # noqa: E402
from variants.experiments.common.protocol import load_material, save_material, MAP_NAMES  # noqa: E402
from variants.experiments.esrgan_map_degradation.core import assemble                 # noqa: E402
from variants.experiments.esrgan_map_degradation.core import dataset as D             # noqa: E402
from variants.experiments.esrgan_map_degradation.core.model import MAPS               # noqa: E402


@torch.no_grad()
def infer(G, lr_enc, device, tile=0):
    x = D.to_batch([lr_enc], device)
    if not tile:
        return {m: v[0] for m, v in G(x).items()}
    h = x[MAPS[0]].shape[-1]
    if h % tile:
        raise ValueError("LR size %d not divisible by tile %d" % (h, tile))
    out = {m: torch.zeros(3, h * 4, h * 4, device=device) for m in MAPS}
    for y in range(0, h, tile):
        for x0 in range(0, h, tile):
            sub = {m: v[..., y:y + tile, x0:x0 + tile] for m, v in x.items()}
            sr = G(sub)
            for m in MAPS:
                out[m][:, 4 * y:4 * (y + tile), 4 * x0:4 * (x0 + tile)] = sr[m][0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="e.g. exp2/2-0_base_scratch (under /expdata/runs)")
    ap.add_argument("--ckpt", help="default: newest")
    ap.add_argument("--set", default="eval12")
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--band", type=int, default=4)
    ap.add_argument("--lights", type=int, default=16)
    args = ap.parse_args()

    run = os.path.join(paths.RUNS, args.run)
    ckpts = sorted(glob.glob(os.path.join(run, "ckpt", "iter_*.pt")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    ck = args.ckpt or (ckpts[-1] if ckpts else None)
    if not ck:
        raise SystemExit("no checkpoint in " + run)
    dev = torch.device("cuda:0")
    st = torch.load(ck, map_location=dev)
    mc = st["config"]["model"]
    G = assemble.generator(mc["arch"], mc["input"], mc.get("num_block", 23), mc.get("num_feat", 64),
                           mc.get("num_grow_ch", 32)).to(dev).eval()
    G.load_state_dict(st["G"])
    tag = "iter_%d" % st["iter"] + ("_tile%d" % args.tile if args.tile else "")
    out = os.path.join(run, "eval", tag)
    os.makedirs(out, exist_ok=True)

    names = sorted(os.path.basename(os.path.dirname(p)) for p in
                   glob.glob(os.path.join(paths.MATERIALS, args.set, "*", "meta.json")))
    results = {}
    for name in names:
        mdir = os.path.join(paths.MATERIALS, args.set, name)
        lr, gt = load_material(os.path.join(mdir, "LR")), load_material(os.path.join(mdir, "GT"))
        pred = D.decode_np(infer(G, D.encode_np(lr), dev, args.tile))
        save_material(pred, os.path.join(out, name))
        r = M.evaluate(pred, gt, dev, maps=MAP_NAMES, border=0, n_lights=args.lights)
        if args.tile:
            r["seam"] = M.masked_pixel_metrics(pred, gt, M.band_mask(gt["basecolor"].shape[0],
                                                                     4 * args.tile, args.band))
        results[name] = r
        mp = r["maps"]
        print("  %-28s bc %.2fdB  n %.2fdeg  r %.4f  m %.4f  h %.4f  FLIP %.4f" % (
            name, mp["basecolor"]["psnr"] or np.inf, mp["normal"]["angle_mean"], mp["roughness"]["mae"],
            mp["metallic"]["mae"], mp["height"]["mae"], r["render"]["flip"]), flush=True)

    def mean(path):
        vals = []
        for r in results.values():
            v = r
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    summary = {
        "basecolor": {k: mean(("maps", "basecolor", k)) for k in ("psnr", "ssim", "lpips", "deltaE00")},
        "normal": {k: mean(("maps", "normal", k)) for k in ("angle_mean", "within5", "within10")},
        "roughness": {"mae": mean(("maps", "roughness", "mae"))},
        "metallic": {k: mean(("maps", "metallic", k)) for k in ("mae", "acc01")},
        "height": {k: mean(("maps", "height", k)) for k in ("mae", "mae_mean_aligned", "grad_err")},
        "render": {k: mean(("render", k)) for k in ("flip", "lpips")},
    }
    json.dump({"ckpt": ck, "iter": st["iter"], "tile": args.tile, "summary": summary,
               "materials": results}, open(os.path.join(out, "results.json"), "w"), indent=2)
    lines = ["# %s @ %s\n" % (args.run, tag), "| map | metric | mean over %d |" % len(results), "|---|---|---|"]
    for mp, d in summary.items():
        for k, v in d.items():
            lines.append("| %s | %s | %s |" % (mp, k, "—" if v is None else "%.4f" % v))
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
