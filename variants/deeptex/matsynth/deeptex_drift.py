#!/usr/bin/env python3
"""How far each checkpoint's output colour sits from its target, checkpoint by checkpoint.

Runs in the deeptex container:

    docker compose run --rm deeptex python /workspace/variants/deeptex/matsynth/deeptex_drift.py \\
        --cfg      /workspace/variants/deeptex/matsynth/esrgan_matsynth.yaml \\
        --ckpt-dir /workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k \\
        --pairs    /workspace/runs/deeptex/datasets/matsynth/baseColor \\
        --out      /workspace/runs/deeptex/drift/matsynth_all

For every checkpoint in --ckpt-dir (each ckpt-N.index, in step order - point it at
checkpoints_all/ from run_deeptex.ps1 -KeepAllCheckpoints to get all eight), on

  train  --n seeded pairs from --pairs/{HR,LRx4} - the model's OWN training data
  eval   the evaluation materials' --eval-map (runs/matsynth/<m>/SVBRDF -> GT)

it reports mean(SR) - mean(HR) per RGB channel in 8-bit levels, their mean
(brightness), the green cast G - (R+B)/2, and the mean per-image largest offset.
LR and HR have the same channel means by construction (area decimation), so any
offset here is something the network added, and the train column rules out the
evaluation pipeline as its source.

WHY THIS EXISTS: the recipe (no pre-training, pixel weight 0.01, a saturated
discriminator) lets the global colour wander between checkpoints, and the MatSynth
arm adds a green cast that grows (DEEPTEX-DEVIATIONS 0.9). The ratio ablations
(0.10) are judged on exactly these numbers, so they are measured by one script
rather than by whatever a scratch file did that day.

--sheet-material <name> also writes, per checkpoint, that material's five maps
plus the render as ONE row (deeptex_compare.py's encoding and collapse, the
committed sheets' cell rendering) - the checkpoint's output only, no comparison
columns.
"""
import argparse
import glob
import json
import os
import random
import re
import sys

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.common import paths  # noqa: E402
SUBMODULE = paths.use_deeptex()

import tensorflow as tf  # noqa: E402

from modules.models import RRDB_Model  # noqa: E402
from modules.utils import load_yaml, set_memory_growth  # noqa: E402

# Bound as `deeptex_compare`: the old file read JOBS (all six), load_as_rgb01,
# save_sr(..., "mean") and newest_run from deeptex_compare.py, which in this tree is
# the infer core plus the all-maps arm. eval_arms re-exports all four.
from variants.deeptex.matsynth import eval_arms as deeptex_compare  # noqa: E402
from variants.deeptex.infer import deeptex_sr  # noqa: E402
from variants.common import map_encode  # noqa: E402
from variants.compare import matup_montage  # noqa: E402
from variants.compare.matup_overview import fit  # noqa: E402

SHEET_MAPS = [j[0] for j in deeptex_compare.JOBS]   # baseColor ... height, render


def rgb8(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise SystemExit("cannot read " + path)
    return map_encode.to_rgb8(a)


def checkpoints(ckpt_dir):
    found = []
    for idx in glob.glob(os.path.join(ckpt_dir, "ckpt-*.index")):
        m = re.search(r"ckpt-(\d+)\.index$", idx)
        if m:
            found.append((int(m.group(1)), idx[:-len(".index")]))
    if not found:
        raise SystemExit("no ckpt-*.index under %s" % ckpt_dir)
    return [p for _, p in sorted(found)]


def offsets(model, items):
    """items: [(lr_path, hr_path)] -> per-image mean(SR) - mean(HR), RGB, 8-bit levels."""
    out = []
    for lp, hp in items:
        lr = rgb8(lp).astype(np.float32) / 255.0
        sr = np.clip(deeptex_sr.super_resolve(model, lr, 4, 0, 16), 0.0, 1.0) * 255.0
        hr = rgb8(hp).astype(np.float64)
        out.append(sr.reshape(-1, 3).mean(0) - hr.reshape(-1, 3).mean(0))
    return np.array(out)


def summarise(b):
    m = b.mean(0)
    return {"n": int(len(b)), "R": float(m[0]), "G": float(m[1]), "B": float(m[2]),
            "brightness": float(m.mean()), "cast": float(m[1] - (m[0] + m[2]) / 2.0),
            "max_abs_offset": float(np.abs(b).max(1).mean())}


def write_sheet(path, title, label, outputs):
    cell, pad, left, top, head = 320, 6, 150, 32, 34
    w = left + len(SHEET_MAPS) * (cell + pad) + pad
    s = np.full((top + head + cell + pad, w, 3), 18, np.uint8)
    cv2.putText(s, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for c, m in enumerate(SHEET_MAPS):
        x = left + c * (cell + pad)
        cv2.putText(s, m, (x + 4, top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (210, 210, 210), 1, cv2.LINE_AA)
        a = outputs.get(m)
        s[top + head:top + head + cell, x:x + cell] = (
            matup_montage.to_display(a, cell) if a is not None else matup_montage.placeholder(cell, ["missing"]))
    for i, t in enumerate(label):
        cv2.putText(s, t, (10, top + head + cell // 2 - 8 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    fit(t, left - 18, 0.62), (210, 210, 210), 1, cv2.LINE_AA)
    if not cv2.imwrite(path, s):
        raise RuntimeError("cv2.imwrite failed for " + path)


def sheet_outputs(model, mat_dir, out_dir):
    """deeptex_compare.py's own path for one material, all six inputs."""
    name = os.path.basename(mat_dir.rstrip("/"))
    run_dir = deeptex_compare.newest_run(mat_dir, name, "")
    os.makedirs(out_dir, exist_ok=True)
    got = {}
    for key, where, fname, kind in deeptex_compare.JOBS:
        src = os.path.join(mat_dir, "SVBRDF", fname) if where == "svbrdf" else \
            (os.path.join(run_dir, fname) if run_dir else "")
        if not src or not os.path.exists(src):
            continue
        lr, _ = deeptex_compare.load_as_rgb01(src)
        sr = deeptex_sr.super_resolve(model, lr, 4, 0, 16)
        p = os.path.join(out_dir, key + ".png")
        deeptex_compare.save_sr(p, sr, kind, "mean")
        got[key] = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt-dir", required=True,
                    help="a checkpoints/<sub_name> or checkpoints_all/<sub_name> directory")
    ap.add_argument("--pairs", required=True, help="a folder holding HR/ and LRx4/")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-root", default="/workspace/runs/matsynth")
    ap.add_argument("--eval-map", default="baseColor", help="file stem under SVBRDF/ and GT/")
    ap.add_argument("--sheet-material", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_memory_growth()
    cfg = load_yaml(args.cfg)
    model = RRDB_Model(None, cfg["ch_size"], cfg["network_G"])

    names = sorted(f for f in os.listdir(os.path.join(args.pairs, "HR")) if f.endswith(".png"))
    pick = random.Random(args.seed).sample(names, min(args.n, len(names)))
    train = [(os.path.join(args.pairs, "LRx4", n), os.path.join(args.pairs, "HR", n)) for n in pick]
    mats = sorted(d for d in glob.glob(os.path.join(args.eval_root, "*"))
                  if os.path.isdir(os.path.join(d, "SVBRDF")))
    ev = [(os.path.join(d, "SVBRDF", args.eval_map + ".png"), os.path.join(d, "GT", args.eval_map + ".png"))
          for d in mats]
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for prefix in checkpoints(args.ckpt_dir):
        deeptex_sr.check_checkpoint(args.ckpt_dir, prefix)
        tf.train.Checkpoint(model=model).restore(prefix).expect_partial()
        step = int(tf.train.load_checkpoint(prefix).get_tensor("step/.ATTRIBUTES/VARIABLE_VALUE"))
        ck = os.path.basename(prefix)
        row = {"checkpoint": ck, "step": step,
               "train": summarise(offsets(model, train)), "eval": summarise(offsets(model, ev))}
        if args.sheet_material:
            mat_dir = os.path.join(args.eval_root, args.sheet_material)
            outs = sheet_outputs(model, mat_dir, os.path.join(args.out, "outputs", ck, args.sheet_material))
            when = "%dk" % (step // 1000) if step % 1000 == 0 else str(step)
            write_sheet(os.path.join(args.out, "%s_%s_%s.png" % (args.sheet_material, ck, when)),
                        "%s    %s  step %d" % (args.sheet_material, ck, step), [ck, when], outs)
        rows.append(row)
        t, e = row["train"], row["eval"]
        print("  %-8s step %6d | train R %+6.2f G %+6.2f B %+6.2f cast %+5.2f | eval R %+6.2f G %+6.2f B %+6.2f cast %+5.2f"
              % (ck, step, t["R"], t["G"], t["B"], t["cast"], e["R"], e["G"], e["B"], e["cast"]))

    with open(os.path.join(args.out, "drift.json"), "w") as f:
        json.dump({"cfg": args.cfg, "ckpt_dir": args.ckpt_dir, "pairs": args.pairs,
                   "n": len(train), "seed": args.seed, "eval_root": args.eval_root,
                   "eval_map": args.eval_map, "eval_materials": len(ev), "checkpoints": rows}, f, indent=1)
    lines = ["# Colour drift per checkpoint", "",
             "mean(SR) - mean(HR) per channel, 8-bit levels; cast = G - (R+B)/2; max|off| = mean",
             "over images of the largest channel offset. train = %d seeded pairs from `%s` (seed %d);"
             % (len(train), args.pairs, args.seed),
             "eval = %d materials' `%s`. Checkpoints from `%s`." % (len(ev), args.eval_map, args.ckpt_dir), "",
             "| checkpoint | step | train R | G | B | brightness | cast | max\\|off\\| "
             "| eval R | G | B | brightness | cast | max\\|off\\| |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t, e = r["train"], r["eval"]
        lines.append("| %s | %d | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f | %.2f | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f | %.2f |"
                     % (r["checkpoint"], r["step"], t["R"], t["G"], t["B"], t["brightness"], t["cast"],
                        t["max_abs_offset"], e["R"], e["G"], e["B"], e["brightness"], e["cast"],
                        e["max_abs_offset"]))
    with open(os.path.join(args.out, "drift.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("%d checkpoint(s) -> %s" % (len(rows), os.path.join(args.out, "drift.md")))


if __name__ == "__main__":
    main()
