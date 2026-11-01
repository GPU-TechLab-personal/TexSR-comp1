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
from variants.util import paths  # noqa: E402
SUBMODULE = paths.use_deeptex()

import tensorflow as tf  # noqa: E402

from modules.models import RRDB_Model  # noqa: E402
from modules.utils import load_yaml, set_memory_growth  # noqa: E402

# Bound as `deeptex_compare`: the old file read JOBS (all six), load_as_rgb01,
# save_sr(..., "mean") and newest_run from deeptex_compare.py, which in this tree is
# the infer core plus the all-maps arm. eval_arms re-exports all four.
from variants.deeptex.matsynth import eval_arms as deeptex_compare  # noqa: E402
from variants.deeptex.infer import deeptex_sr  # noqa: E402
from variants.util.report import drift  # noqa: E402

def checkpoints(ckpt_dir):
    found = []
    for idx in glob.glob(os.path.join(ckpt_dir, "ckpt-*.index")):
        m = re.search(r"ckpt-(\d+)\.index$", idx)
        if m:
            found.append((int(m.group(1)), idx[:-len(".index")]))
    if not found:
        raise SystemExit("no ckpt-*.index under %s" % ckpt_dir)
    return [p for _, p in sorted(found)]


# offsets / summarise / sheet / tables: variants/util/report/drift.py (any method).


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
    predict = deeptex_sr.predictor(model)

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
               "train": drift.summarise(drift.offsets(predict, train)),
               "eval": drift.summarise(drift.offsets(predict, ev))}
        if args.sheet_material:
            mat_dir = os.path.join(args.eval_root, args.sheet_material)
            outs = drift.sheet_outputs(predict, mat_dir, os.path.join(args.out, "outputs", ck, args.sheet_material))
            when = "%dk" % (step // 1000) if step % 1000 == 0 else str(step)
            drift.write_sheet(os.path.join(args.out, "%s_%s_%s.png" % (args.sheet_material, ck, when)),
                        "%s    %s  step %d" % (args.sheet_material, ck, step), [ck, when], outs)
        rows.append(row)
        drift.print_row(row)

    drift.write_report(args.out, {"cfg": args.cfg, "ckpt_dir": args.ckpt_dir,
                                  "pairs": args.pairs, "n": len(train), "seed": args.seed,
                                  "eval_root": args.eval_root, "eval_map": args.eval_map},
                       rows, len(train), len(ev), args.pairs, args.seed, args.eval_map,
                       args.ckpt_dir)


if __name__ == "__main__":
    main()
