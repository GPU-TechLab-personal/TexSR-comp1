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

from variants.util import map_encode  # noqa: E402
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


def load_as_rgb01(path):
    """Any of the five map encodings -> HxWx3 float32 in [0,1], plus provenance.

    One cv2 path for everything, because the inputs are genuinely heterogeneous:
    8-bit RGB (baseColor, normal), 8-bit grey (roughness, metallic), 16-bit grey
    (height) and 16-bit RGB (render_lr). deeptex_sr.read_rgb uses the TF decoder
    and cannot express the 16-bit cases.

    cv2 hands back BGR, and the network was trained on RGB (modules/dataset.py:12-13
    decodes with tf.image.decode_png) - that mismatch is exactly test.py's bug
    (:57,:68,:70), so the channel reversal is explicit and load-bearing. For the
    grey maps it is a no-op because all three channels are equal.

    The encoding itself is map_encode.to_rgb8, which matsynth_make_pairs.py also
    uses to build the MatSynth training pairs - so train and eval cannot drift.
    """
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError("cv2.imread returned None: %s" % path)
    src = {"dtype": str(arr.dtype),
           "channels": 1 if arr.ndim == 2 else int(arr.shape[2])}
    try:
        rgb = map_encode.to_rgb8(arr)
    except ValueError as e:
        raise RuntimeError("%s in %s" % (e, path))
    return rgb.astype(np.float32) / 255.0, src


def save_colour(path, sr):
    """A 3-channel map back to a 3-channel PNG. Returns (shape, None): no spread."""
    u8 = np.rint(np.clip(sr, 0.0, 1.0) * 255.0).astype(np.uint8)
    out = u8[:, :, ::-1]          # RGB -> BGR for cv2
    if not cv2.imwrite(path, out, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError("cv2.imwrite failed for " + path)
    return list(out.shape), None


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
    wanted = None if maps == "all" else set(maps.split(","))
    jobs = [j for j in all_jobs if wanted is None or j[0] in wanted]
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

    mats = [os.path.join(args.root, args.material)] if args.material else \
        sorted(d for d in glob.glob(os.path.join(args.root, "*"))
               if os.path.isdir(os.path.join(d, "SVBRDF")))
    if not mats:
        deeptex_sr.die(2, "no materials under %s" % args.root)

    os.makedirs(args.out, exist_ok=True)
    done, skipped = [], []

    for mat_dir in mats:
        name = os.path.basename(mat_dir.rstrip("/"))
        run_dir = newest_run(mat_dir, name, args.run)
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        wrote = {}

        for key, where, fname, kind in jobs:
            if where == "svbrdf":
                src = os.path.join(mat_dir, "SVBRDF", fname)
            elif run_dir:
                src = os.path.join(run_dir, fname)
            else:
                skipped.append("%s/%s (no MatUp run)" % (name, key))
                continue
            if not os.path.exists(src):
                skipped.append("%s/%s (%s missing)" % (name, key, os.path.basename(src)))
                continue
            try:
                lr, prov = load_as_rgb01(src)
                sr = deeptex_sr.super_resolve(model, lr, scale, args.tile_size, args.overlap)
                if not np.isfinite(sr).all():
                    deeptex_sr.die(4, "non-finite output on %s/%s" % (name, key))
                out_shape, spread = save(os.path.join(out_dir, key + ".png"), sr, kind)
            except tf.errors.ResourceExhaustedError as e:
                deeptex_sr.die(10, "OOM on %s/%s - lower --tile-size\n%s" % (name, key, e))
            wrote[key] = {"src": src, "kind": kind, "source_encoding": prov,
                          "lr_shape": list(lr.shape), "sr_shape": out_shape}
            if spread:
                wrote[key]["channel_spread"] = spread

        with open(os.path.join(out_dir, "sr_manifest.json"), "w") as f:
            json.dump({"material": name, "checkpoint": prefix, "step": step,
                       "sub_name": cfg["sub_name"], "scale": scale,
                       "matup_run": os.path.basename(run_dir) if run_dir else None,
                       "scalar_collapse": collapse[0],
                       "collapse_weights": collapse[1],
                       "outputs": wrote}, f, indent=2)
        done.append(name)
        sp = [(k, v["channel_spread"]["mean_255"]) for k, v in sorted(wrote.items())
              if "channel_spread" in v]
        extra = ("   ch-spread " + " ".join("%s %.1f" % kv for kv in sp)) if sp else ""
        print("  %-28s %s%s" % (name, ", ".join(sorted(wrote)) or "nothing", extra))

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
