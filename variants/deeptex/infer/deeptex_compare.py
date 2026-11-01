#!/usr/bin/env python3
"""Run DeepTextures over the LR inputs MatUp's comparison already provides.

Runs in the deeptex container:

    python /workspace/variants/deeptex/infer/deeptex_compare.py \\
        --cfg /workspace/variants/deeptex/paper/esrgan_paper.yaml \\
        --ckpt-dir /workspace/runs/deeptex/paper/checkpoints/esrgan_40k \\
        --out /workspace/runs/deeptex/eval/paper

Bridges a trained checkpoint into the existing compare chain, so
variants/deeptex/paper/deeptex_postprocess.py --deeptex-root can add a fifth column.

This is the shared core of the old scripts/deeptex/deeptex_compare.py: the two maps
an RGB network can be scored on without inventing anything - baseColor (from the
material's LR set) and render (MatUp's render_lr.png) - which is what the old
script did by default (--maps baseColor,render). The all-maps mode with its 3->1
collapse (old --maps all --scalar-collapse) is an arm of its own:
variants/deeptex/matsynth/eval_arms.py, which runs through run() below with its
own job list and writer.

The old docstring's scope discussion (which maps the paper covers, what blocks
each one, why 128 -> 512 works with size=None) is unchanged and lives in
scripts/deeptex/deeptex_compare.py:7-64 and docs/DEEPTEX-DEVIATIONS.md 0.6.

Outputs: <out>/<material>/{baseColor,render}.png and sr_manifest.json, same names,
same bytes, same manifest keys as before.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.util import paths  # noqa: E402
SUBMODULE = paths.use_deeptex()

import tensorflow as tf  # noqa: E402

from modules.models import RRDB_Model  # noqa: E402
from modules.utils import load_yaml, set_memory_growth  # noqa: E402

from variants.util import arm_output  # noqa: E402
from variants.matup.baseline.matup_outputs import newest_run  # noqa: E402,F401
from variants.deeptex.infer import deeptex_sr  # noqa: E402

# name -> (source, kind). "svbrdf" sources come from the material's LR set, "run"
# sources from the MatUp output folder. Both are 3-channel maps the network maps
# straight back to 3 channels: no collapse is involved.
JOBS = [
    ("baseColor", "svbrdf", "baseColor.png", "colour"),
    ("render",    "run",    "render_lr.png", "colour"),
]

# What every sr_manifest.json has recorded under scalar_collapse /
# collapse_weights: the old --scalar-collapse default, written even when no
# 1-channel map was run. Kept so the manifest is the old one; nothing here
# applies it (matsynth/eval_arms.py is where a collapse happens).
RECORDED_COLLAPSE = ("mean", (1 / 3.0, 1 / 3.0, 1 / 3.0))


# The writer every SR arm shares (numpy + cv2, no TF): variants/util/arm_output.py,
# whose docstring is the evaluation-folder contract.
load_as_rgb01 = arm_output.load_as_rgb01
save_colour = arm_output.save_colour


def save_job(path, sr, kind):
    """This entry's writer: every one of its jobs is a colour map."""
    if kind != "colour":
        raise ValueError("deeptex_compare writes colour maps only, got %r "
                         "(the 1-channel maps are matsynth/eval_arms.py)" % kind)
    return save_colour(path, sr)


def parser(jobs, default_maps, description=__doc__):
    ap = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--root", default="/workspace/runs/matsynth")
    ap.add_argument("--out", required=True, help="e.g. /workspace/runs/deeptex/eval/<tag>")
    ap.add_argument("--material", default="")
    ap.add_argument("--maps", default=default_maps,
                    help="comma-separated subset of %s, or 'all'"
                         % ",".join(j[0] for j in jobs))
    ap.add_argument("--run", default="", help="MatUp output stamp; default is the newest")
    ap.add_argument("--tile-size", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--gpu", default="0")
    return ap


def select(all_jobs, maps):
    jobs = arm_output.select(all_jobs, maps)
    if not jobs:
        deeptex_sr.die(2, "--maps matched nothing")
    return jobs


def run(args, jobs, save, collapse):
    """Restore the checkpoint, super-resolve every job of every material, write
    <out>/<material>/<map>.png + sr_manifest.json.

    save(path, sr, kind) -> (shape, channel_spread or None); collapse = (name,
    weights) is what the manifest records.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_memory_growth()
    cfg = load_yaml(args.cfg)
    scale = cfg["scale"]

    model = RRDB_Model(None, cfg["ch_size"], cfg["network_G"])
    ckpt_dir = args.ckpt_dir or os.path.join(
        cfg.get("checkpoint_root", "/workspace/runs/deeptex/checkpoints"), cfg["sub_name"])
    prefix, step, _ = deeptex_sr.check_checkpoint(ckpt_dir, args.ckpt)
    before = [v.numpy().copy() for v in model.trainable_variables]
    status = tf.train.Checkpoint(model=model).restore(prefix)
    status.expect_partial()
    status.assert_existing_objects_matched()
    if all(np.array_equal(b, v.numpy())
           for b, v in zip(before, model.trainable_variables)):
        deeptex_sr.die(3, "restoring %s changed nothing" % prefix)

    mats = arm_output.materials(args.root, args.material)
    if not mats:
        deeptex_sr.die(2, "no materials under %s" % args.root)

    predict = deeptex_sr.predictor(model)

    def sr_fn(lr, where):
        try:
            return arm_output.super_resolve(predict, lr, scale, args.tile_size, args.overlap)
        except tf.errors.ResourceExhaustedError as e:
            deeptex_sr.die(10, "OOM on %s - lower --tile-size\n%s" % (where, e))

    done, skipped = arm_output.write_arm(
        mats, args.out, jobs, sr_fn, save, collapse,
        {"checkpoint": prefix, "step": step, "sub_name": cfg["sub_name"], "scale": scale},
        run=args.run, die=deeptex_sr.die)

    print("%d material(s), %d map(s) each -> %s  (checkpoint step %s)"
          % (len(done), len(jobs), args.out, step))
    for s in skipped[:20]:
        print("  [skip] %s" % s)
    if len(skipped) > 20:
        print("  [skip] ... and %d more" % (len(skipped) - 20))


def main():
    args = parser(JOBS, "baseColor,render").parse_args()
    run(args, select(JOBS, args.maps), save_job, RECORDED_COLLAPSE)


if __name__ == "__main__":
    main()
