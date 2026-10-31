#!/usr/bin/env python3
"""
실험 1 (간단 지표): the cross-light inconsistency C of an upsampler S.

    D_w = S(R_w) - bilinear_up(R_w)        per light w, HR display space
    C   = mean over pixels (and RGB) of std_w(D_w)

R_w are MatUp's own LR renders (100 Fibonacci lights, AA 2x2) and S sees them
exactly as MatUp's upsampling step does (sr_cache.py). The bilinear reference
is taken from the same 8-bit input S saw, so D is only what S adds.

    python variants/experiments/exp1_inconsistency/run_C.py --pilot --s swinir bilinear
    python variants/experiments/exp1_inconsistency/run_C.py --s swinir sdx4 srflow

Also recorded, so C can be read against them:
    D_rms          overall detail energy S adds
    consistent     mean_x |mean_w D_w| - the part every light agrees on
    C_rel          C / D_rms - inconsistency as a fraction of what S adds
and the material factors of the analysis (GT at 512^2 and LR):
    normal_grad_var, normal_tilt_deg, roughness_mean, basecolor_lum_std,
    basecolor_rms_contrast (the control)

Output: /expdata/runs/exp1/C/<S tag>/<material>.json (+ C map .npy, float16)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.experiments.common import cli, paths                                  # noqa: E402
from variants.experiments.common.protocol import load_material, srgb_to_linear      # noqa: E402
from variants.experiments.common.sr_cache import ensure_sr_renders, read_lr_renders, read_sr_renders  # noqa: E402
from variants.experiments.common.upscalers import bilinear                          # noqa: E402


def material_factors(mat):
    n = mat["normal"].astype(np.float64)
    gy, gx = np.gradient(n, axis=(0, 1))
    lin = srgb_to_linear(mat["basecolor"])
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    return {"normal_grad_var": float((gx ** 2 + gy ** 2).sum(-1).mean()),
            "normal_tilt_deg": float(np.degrees(np.arccos(np.clip(n[..., 2], -1, 1))).mean()),
            "roughness_mean": float(mat["roughness"].mean()),
            "roughness_std": float(mat["roughness"].std()),
            "metallic_mean": float(mat["metallic"].mean()),
            "basecolor_lum_std": float(lum.std()),
            "basecolor_rms_contrast": float(lum.std() / max(lum.mean(), 1e-6))}


def measure(lr_dir, sr_dir, border, device):
    x = read_lr_renders(lr_dir).to(device)          # (L, 3, h, w), what S saw
    y = read_sr_renders(sr_dir).to(device)          # (L, 3, 4h, 4w)
    d = y - bilinear(x)
    if border:
        d = d[..., border:-border, border:-border]
    std_w = d.std(dim=0, unbiased=False)            # (3, H, W)
    rms = float(torch.sqrt((d ** 2).mean()))
    c = float(std_w.mean())
    out = {"C": c, "C_per_channel": std_w.mean(dim=(1, 2)).tolist(),
           "D_rms": rms, "consistent": float(d.mean(dim=0).abs().mean()),
           "C_rel": c / rms if rms > 0 else None,
           "per_light_D_rms": torch.sqrt((d ** 2).mean(dim=(1, 2, 3))).tolist()}
    return out, std_w.mean(0).cpu().numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_material_args(ap)
    cli.add_s_args(ap, default=("swinir", "bilinear"))
    ap.add_argument("--lights", type=int, default=100)
    ap.add_argument("--border", type=int, default=8,
                    help="HR px dropped at each edge (SwinIR's mirror padding / MatUp's valid region)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dev = cli.device()
    names = cli.resolve_materials(args)
    print("%d materials x S=%s" % (len(names), args.s))
    for up in cli.iter_upscalers(args, dev):
        out_root = os.path.join(paths.RUNS, "exp1", "C", up.tag)
        os.makedirs(out_root, exist_ok=True)
        for name in names:
            out = os.path.join(out_root, name + ".json")
            if os.path.isfile(out) and not args.force:
                print("  %-12s %-30s cached" % (up.tag, name))
                continue
            t0 = time.perf_counter()
            mat_dir = os.path.join(paths.MATERIALS, args.set, name)
            lr_dir, sr_dir = ensure_sr_renders(os.path.join(mat_dir, "LR"),
                                               os.path.join(paths.SR_CACHE, args.set, name),
                                               up, dev, args.lights)
            stats, cmap = measure(lr_dir, sr_dir, args.border, dev)
            rec = {"material": name, "set": args.set, "upscaler": up.config(),
                   "n_lights": args.lights, "border": args.border, **stats,
                   "factors_gt": material_factors(load_material(os.path.join(mat_dir, "GT"))),
                   "factors_lr": material_factors(load_material(os.path.join(mat_dir, "LR"))),
                   "seconds": time.perf_counter() - t0}
            with open(out, "w") as fh:
                json.dump(rec, fh, indent=2)
            np.save(out.replace(".json", "_Cmap.npy"), cmap)
            print("  %-12s %-30s C %.5f  D_rms %.5f  C_rel %.3f  (%.1fs)" % (
                up.tag, name, stats["C"], stats["D_rms"], stats["C_rel"] or 0, rec["seconds"]))


if __name__ == "__main__":
    main()
