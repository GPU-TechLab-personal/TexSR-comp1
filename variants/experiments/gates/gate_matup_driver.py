#!/usr/bin/env python3
"""
Gate: is variants/experiments/matup_driver/ still MatUp?

    python variants/experiments/gates/gate_matup_driver.py [--material acg_wood_floor_060]

Three comparisons on one eval12 material, S = SwinIR:
  1. LR renders: MatUp's render_lowres.main_render into a scratch dir vs the
     sr_cache copy                                     -> must be byte-identical
  2. SR renders: MatUp's sr_swinir.main_upsample_swinir vs sr_cache
                                                       -> must be byte-identical
  3. The optimisation: the real `python main.py` (CUDA graphs, its own
     rmtree) vs the eager driver. Not bit-identical by construction (kernel
     order, RNG consumption); reported as the metric gap vs GT and the direct
     map difference, so the size of "eager != graph" is a measured number.
~5 min.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.experiments.common import cli, paths, protocol                  # noqa: E402
from variants.experiments.common import metrics as M                          # noqa: E402
from variants.experiments.common.matup_runs import run_one, BORDER, CLAIMED  # noqa: E402
from variants.experiments.common.sr_cache import lr_render_dir, sr_dir        # noqa: E402
from variants.experiments.common.upscalers import get                         # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print("  %-52s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILS.append(name)


def same_pngs(a, b):
    fa, fb = sorted(glob.glob(a + "/*.png")), sorted(glob.glob(b + "/*.png"))
    if len(fa) != len(fb) or not fa:
        return False, "%d vs %d files" % (len(fa), len(fb))
    worst = 0
    for x, y in zip(fa, fb):
        ia, ib = cv2.imread(x, cv2.IMREAD_UNCHANGED), cv2.imread(y, cv2.IMREAD_UNCHANGED)
        worst = max(worst, int(np.abs(ia.astype(np.int64) - ib.astype(np.int64)).max()))
    return worst == 0, "%d files, max abs diff %d" % (len(fa), worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="acg_wood_floor_060")
    ap.add_argument("--set", default="eval12")
    args = ap.parse_args()

    dev = cli.device()
    mat = os.path.join(paths.MATERIALS, args.set, args.material)
    lr_in = os.path.join(mat, "LR") + "/"
    scratch = os.path.join(paths.EXPDATA, "gates", "matup_driver", args.material)
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch)

    up = get("swinir", dev)
    print("[our pipeline] eager driver, variant orig")
    ours = run_one(args.set, args.material, up, dev, variant="orig", force=True)
    cache = os.path.join(paths.SR_CACHE, args.set, args.material)

    print("[MatUp stages 1-2] render_lowres + sr_swinir, scratch dirs")
    sys.path.insert(0, paths.MATUP)
    os.chdir(paths.MATUP)
    import argparse as _a
    from render_lowres import main_render
    from sr_swinir import main_upsample_swinir
    a = _a.Namespace(input_mat=lr_in, render_dir=scratch + "/lr/", upsamplings_dir=scratch + "/up/",
                     crop_size=128, AA_upscale=2, nb_lights=100, disp_val=0.01,
                     light_falloff=False, opengl_normals=True, material_size=128, up_batch_size=10)
    main_render(a)
    main_upsample_swinir(a)
    ok, d = same_pngs(scratch + "/lr", lr_render_dir(cache, 100))
    check("LR renders byte-identical to MatUp's", ok, d)
    ok, d = same_pngs(scratch + "/up", sr_dir(cache, "swinir", 100))
    check("SwinIR renders byte-identical to MatUp's", ok, d)

    print("[MatUp end-to-end] python main.py (CUDA graphs)")
    cmd = [sys.executable, "main.py", "--upscaler", "swinir", "--ogl_normals",
           "--input_mat", lr_in, "--render_dir", scratch + "/e2e_lr/",
           "--upsamplings_dir", scratch + "/e2e_up/", "--output_folder", scratch + "/e2e_out/"]
    p = subprocess.run(cmd, cwd=paths.MATUP, capture_output=True, text=True)
    check("main.py exits 0", p.returncode == 0, "" if p.returncode == 0 else p.stderr[-800:])
    runs = sorted(glob.glob(scratch + "/e2e_out/*/"))
    if not runs:
        print(p.stdout[-2000:])
        return 1
    theirs_dir = runs[-1]

    # MatUp's own PNGs -> canonical (DirectX normal back to OpenGL, 8px border off)
    raw = {k: cv2.imread(os.path.join(theirs_dir, f.replace(".png", "_opt.png")), cv2.IMREAD_UNCHANGED)
           for k, f in protocol.MAP_FILES.items() if k != "height"}
    raw["height"] = cv2.imread(os.path.join(lr_in, "height.png"), cv2.IMREAD_UNCHANGED)
    raw["height"] = cv2.resize(raw["height"], (512, 512), interpolation=cv2.INTER_LINEAR)
    th = protocol.from_raw(raw)
    th["normal"][..., 1] = -th["normal"][..., 1]          # MatUp writes DirectX
    th = {k: protocol.crop_border(v, BORDER) for k, v in th.items()}
    gt = {k: protocol.crop_border(v, BORDER) for k, v in protocol.load_material(os.path.join(mat, "GT")).items()}
    theirs = M.evaluate(th, gt, dev, maps=CLAIMED, border=0, n_lights=16, full_size=512)

    # the driver writes straight into maps/ (no main_optim timestamp folder)
    ours_run = os.path.join(paths.RUNS, "matup", args.set, "swinir", args.material, "orig", "maps")
    ou = {k: cv2.imread(os.path.join(ours_run, f.replace(".png", "_opt.png")), cv2.IMREAD_UNCHANGED)
          for k, f in protocol.MAP_FILES.items() if k != "height"}
    ou["height"] = raw["height"]
    ou = protocol.from_raw(ou)
    ou["normal"][..., 1] = -ou["normal"][..., 1]
    ou = {k: protocol.crop_border(v, BORDER) for k, v in ou.items()}

    print("\n  metric vs GT              main.py      eager driver")
    rows = [("basecolor PSNR", "basecolor", "psnr"), ("basecolor LPIPS", "basecolor", "lpips"),
            ("normal angle (deg)", "normal", "angle_mean"), ("roughness MAE", "roughness", "mae"),
            ("metallic MAE", "metallic", "mae")]
    for label, m, k in rows:
        print("  %-24s %10.4f   %10.4f" % (label, theirs["maps"][m][k], ours["metrics"]["maps"][m][k]))
    print("  %-24s %10.4f   %10.4f" % ("render FLIP", theirs["render"]["flip"],
                                        ours["metrics"]["render"]["flip"]))
    ang = M.angular_error_deg(th["normal"], ou["normal"])
    bc = np.abs(th["basecolor"] - ou["basecolor"])
    print("\n  main.py vs eager, direct: basecolor |d| mean %.4f max %.4f, normal angle mean %.3f deg"
          % (bc.mean(), bc.max(), ang.mean()))
    gap = abs(theirs["maps"]["normal"]["angle_mean"] - ours["metrics"]["maps"]["normal"]["angle_mean"])
    check("normal-angle gap vs GT < 0.5 deg", gap < 0.5, "%.3f" % gap)
    gap = abs((theirs["maps"]["basecolor"]["psnr"] or 99) - (ours["metrics"]["maps"]["basecolor"]["psnr"] or 99))
    check("basecolor PSNR gap < 0.3 dB", gap < 0.3, "%.3f" % gap)
    print("\n%s" % ("ALL PASS" if not FAILS else "FAILED: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
