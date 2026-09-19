#!/usr/bin/env python3
"""Run DeepTextures over the LR inputs MatUp's comparison already provides.

Runs in the deeptex container. Bridges a trained checkpoint into the existing
compare chain so matup_postprocess.py --deeptex-root can add a fifth row.

SCOPE: which maps get a DeepTextures column.
--------------------------------------------
REVISED 2026-09-19. An earlier version of this docstring gave three reasons for
restricting the run to baseColor and render. Two of them were wrong. They are
corrected here; the record of the error is in docs/DEEPTEX-DEVIATIONS.md 0.6.

The paper covers all five channels - Alonso & Patow, Sci. Rep. (2026) 16:12532,
verified verbatim against the published article: "We have enabled multi-channel
material upscaling, allowing the model to simultaneously process and enhance
various material maps, including displacement, metalness, normals, and
roughness", and Fig. 7 shows all four. So a map-by-map comparison is within the
method's intent.

WHAT ACTUALLY BLOCKS EACH MAP - they are not the same problem:

  normal     3ch GT, 3ch output. NOTHING to collapse. The "a super-resolved
             normal is not unit length" objection is OURS, not the paper's: the
             paper treats a normal map as an image and measures image error.
             This is the EASIEST case, not a hard one.
  roughness  1ch GT, so the montage column needs a 3->1 reduction of an output
             whose three channels differ (conv_last, models.py:109, has three
             untied filters). The repo DOES specify a reduction - but only on
             the METRIC path: test.py:91-94 wraps both sides in rgb2ycbcr(),
             whose only_y=True default (utils.py:340) returns a 1-channel Y, and
             because test.py feeds it cv2's BGR the weights that actually land
             are R/B-swapped BT.601 (utils.py:351,355). There is no rule for
             producing a 1-channel IMAGE. Using test.py's own weights makes the
             choice repo-faithful rather than invented - that is COLLAPSE["paper"].
  metallic   Same as roughness, plus: the GT is a constant 0 in 9 of the 12
             evaluation materials (measured). The column is only informative for
             2119_corten_steel, acg_metal_026 and tc_metal_029.
  height     Blocked for reasons unrelated to channel count: MatUp emits no
             height at all (ours is reconstructed by matup_height.py), so the
             column would compare two different kinds of thing, and 8 bits are
             lost through a uint8 network.

PAPER-FAITHFUL IS NOT AVAILABLE FOR ANY OF THEM. The paper's reported metrics
are delta error and NVIDIA FLIP, and NEITHER has any code in the released
repository - grep finds only the `using_flip` augmentation flag. The published
numbers came from code that was never released, so no choice we make here can
claim to match the paper's. Repo-faithful is the strongest available standard.

THE REAL LIMITATION, worth more than all of the above: the released
data/textures_db is 300 files (200 png + 100 jpg), not the paper's 1,500 (300
textures x 5 channels), and the paper's 1,050-image training split ships
nowhere - the Zenodo archive at 10.5281/zenodo.17495609 is a snapshot of the
same GitHub tag. Measured composition of the 299 that normalise: 262 colour, 26
near-grayscale, 11 normal-map-like. So this checkpoint barely saw a normal or a
roughness map in training. A weak result on those maps is a limit of the
RELEASED DATA, not of the method, and must be reported that way.

WHY 128 -> 512 WORKS AT ALL
---------------------------
cfg['input_size'] = 32 is the TRAINING CROP, not a model constraint. Built with
size=None, RRDB_Model switches its upsample to tf.shape (modules/models.py:98-99)
and is genuinely resolution-independent - which is why test.py:33 builds it that
way too. So the existing 128^2 -> 512^2 pair that MatUp already ran on is reused
verbatim, and both methods see the identical LR bytes.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SUBMODULE = os.environ.get("DEEPTEX_ROOT", "/workspace/external/DeepTextures")
sys.path.insert(0, SUBMODULE)

import tensorflow as tf  # noqa: E402

from modules.models import RRDB_Model  # noqa: E402
from modules.utils import load_yaml, set_memory_growth  # noqa: E402

import deeptex_sr  # noqa: E402

# name -> (source, kind). "svbrdf" sources come from the material's LR set, "run"
# sources from the MatUp output folder. kind decides how the map is squeezed into
# and back out of a 3-channel uint8 network.
JOBS = [
    ("baseColor", "svbrdf", "baseColor.png", "colour"),
    ("normal",    "svbrdf", "normal.png",    "colour"),
    ("roughness", "svbrdf", "roughness.png", "scalar"),
    ("metallic",  "svbrdf", "metallic.png",  "scalar"),
    ("height",    "svbrdf", "height.png",    "scalar"),
    ("render",    "run",    "render_lr.png", "colour"),
]


def load_as_rgb01(path):
    """Any of the five map encodings -> HxWx3 float32 in [0,1], plus provenance.

    One cv2 path for everything, because the inputs are genuinely heterogeneous:
    8-bit RGB (baseColor, normal), 8-bit grey (roughness, metallic), 16-bit grey
    (height) and 16-bit RGB (render_lr). deeptex_sr.read_rgb uses the TF decoder
    and cannot express the 16-bit cases.

    cv2 hands back BGR, and the network was trained on RGB (modules/dataset.py:12-13
    decodes with tf.image.decode_png) - that mismatch is exactly test.py's bug
    (:57,:68,:70), so the channel reversal below is explicit and load-bearing. For
    the grey maps it is a no-op because all three channels are equal.
    """
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError("cv2.imread returned None: %s" % path)
    src = {"dtype": str(arr.dtype),
           "channels": 1 if arr.ndim == 2 else int(arr.shape[2])}

    if arr.dtype == np.uint16:
        # Round, do not shift: a >> 8 biases every value low by up to 1 LSB.
        arr = np.rint(arr / 257.0).clip(0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        raise RuntimeError("unexpected dtype %s in %s" % (arr.dtype, path))

    if arr.ndim == 2:
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.shape[2] == 3:
        rgb = arr[:, :, ::-1]
    elif arr.shape[2] == 4:
        rgb = arr[:, :, 2::-1]
    else:
        raise RuntimeError("unsupported channel count %d in %s" % (arr.shape[2], path))
    return np.ascontiguousarray(rgb).astype(np.float32) / 255.0, src


# Collapsing a scalar map's 3 output channels back to 1. The network saw three
# IDENTICAL channels, but conv_last (modules/models.py:109) has three different
# filters, so the three outputs are NOT identical and the choice is visible.
#
# What upstream does, for reference: test.py never collapses on the WRITE path -
# tensor2img keeps all three and cv2.imwrite emits a 3-channel PNG even for a
# grayscale input. It collapses only for METRICS, via rgb2ycbcr(..., only_y=True)
# (modules/utils.py:340-365), i.e. BT.601 luma. And it feeds that function BGR
# (cv2.imread) when the function reverses channels internally and expects RGB, so
# the R and B weights end up swapped. For a replicated grayscale input that swap
# is invisible - the weights just sum to 219 - but it is visible on the network's
# non-identical OUTPUT channels, which is exactly what we are collapsing.
#
# We must write 1 channel, because the GT and MatUp's column are 1 channel. So the
# question is only which weights.
COLLAPSE = {
    # symmetric - matches the fact that the three input channels were identical
    "mean":  (1 / 3.0, 1 / 3.0, 1 / 3.0),
    # correct BT.601 luma on RGB
    "luma":  (0.299, 0.587, 0.114),
    # what test.py's metric path actually applies, R/B swapped (see above)
    "paper": (0.114, 0.587, 0.299),
}


def save_sr(path, sr, kind, collapse="mean"):
    """Write the super-resolved map back in the shape its column expects.

    Returns (shape, channel_spread) where channel_spread measures how far apart
    the three output channels actually were - if it is ~0 the collapse choice is
    immaterial, and if it is large it has to be stated.
    """
    u8 = np.rint(np.clip(sr, 0.0, 1.0) * 255.0).astype(np.uint8)
    spread = None
    if kind == "scalar":
        f = sr.astype(np.float64)
        d = f.max(axis=2) - f.min(axis=2)
        spread = {"mean_255": float(d.mean() * 255.0),
                  "p99_255": float(np.percentile(d, 99) * 255.0),
                  "max_255": float(d.max() * 255.0)}
        w = COLLAPSE[collapse]
        out = np.rint(np.clip(f[:, :, 0] * w[0] + f[:, :, 1] * w[1] + f[:, :, 2] * w[2],
                              0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        out = u8[:, :, ::-1]          # RGB -> BGR for cv2
    if not cv2.imwrite(path, out, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError("cv2.imwrite failed for " + path)
    return list(out.shape), spread


def newest_run(mat_dir, name, want_stamp):
    """Same selection rule as the rest of the chain: sorted(...)[-1].

    matup_postprocess.py:203-211, matup_render.py:166-172 and matup_montage.py:130
    all do this, so following it keeps the DeepTextures column aligned with the
    MatUp run the other columns came from.
    """
    runs = sorted(d for d in glob.glob(os.path.join(mat_dir, "matup-output-%s" % name, "*"))
                  if os.path.isdir(d))
    if want_stamp:
        runs = [r for r in runs if os.path.basename(r) == want_stamp]
    return runs[-1] if runs else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--root", default="/workspace/runs/matsynth")
    ap.add_argument("--out", required=True, help="e.g. /workspace/runs/deeptex/eval/<tag>")
    ap.add_argument("--material", default="")
    ap.add_argument("--maps", default="baseColor,render",
                    help="comma-separated subset of %s, or 'all'. The default is "
                         "the two maps that need no invented channel collapse"
                         % ",".join(j[0] for j in JOBS))
    ap.add_argument("--run", default="", help="MatUp output stamp; default is the newest")
    ap.add_argument("--scalar-collapse", default="mean", choices=sorted(COLLAPSE),
                    help="how a 1-channel map's 3 network outputs are collapsed back "
                         "to 1. 'paper' reproduces test.py's metric-path weights")
    ap.add_argument("--tile-size", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    wanted = None if args.maps == "all" else set(args.maps.split(","))
    jobs = [j for j in JOBS if wanted is None or j[0] in wanted]
    if not jobs:
        deeptex_sr.die(2, "--maps matched nothing")

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
                out_shape, spread = save_sr(os.path.join(out_dir, key + ".png"),
                                            sr, kind, args.scalar_collapse)
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
                       "scalar_collapse": args.scalar_collapse,
                       "collapse_weights": COLLAPSE[args.scalar_collapse],
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


if __name__ == "__main__":
    main()
