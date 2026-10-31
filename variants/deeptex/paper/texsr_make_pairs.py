#!/usr/bin/env python3
"""Tile normalised images into an exact x4 HR/LRx4 training pair set.

Stage [B] of the DeepTextures data pipeline. Runs in the deeptex container.

Why deeptex and not dataprep, where the rest of our image wrangling lives: this
stage must call imresize_np (modules/utils.py:125), the MATLAB-bicubic resampler
the model is actually trained and evaluated with. The FUNCTION is numpy-only, but
the MODULE is not - modules/utils.py:8,10 do "import tensorflow as tf" and
"from modules.dataset import load_tfrecord_dataset" at module scope, so importing
it under dataprep raises ModuleNotFoundError. Vendoring a copy into scripts/ would
create a second bicubic implementation that can drift from the one training uses,
which is not a thing to carry in a comparison study. So this stage runs where TF
already is, and runs/ is the handoff.

Nothing in external/DeepTextures is modified; its modules are imported and driven
from outside, the same arrangement scripts/matup/matup_render.py:14 established for
matup_egsr.

Four corrections to the donor tiler (external/esrgan-tf2/data/extract_subimages.py):

  1. HR is tiled and LR is derived PER TILE. The donor tiled HR and LR separately
     (:67, :71-74) and trusted the two grids to agree. Deriving LR from each HR
     tile makes hr.shape == lr.shape * scale true BY CONSTRUCTION - which is
     exactly the invariant modules/dataset.py:43-46 asserts and then throws away
     (the tf.Assert return values are discarded, so a violation shows up as a
     misaligned training pair rather than an error).
  2. Small images do not crash. The donor indexes h_space[-1] (:122) on an empty
     arange when h < crop_sz. textures_db runs 64^2..2048^2 with most near 545^2,
     so this would fire constantly.
  3. imap_unordered, not apply_async. The donor's pool (:100) has no
     error_callback, so a worker exception vanishes and the run reports success
     with tiles missing.
  4. Every pair is asserted before it is written.

The holdout is split by SOURCE IMAGE, never by tile: at the default step 240 <
crop 480 the tiles overlap, so a tile-level split leaks pixels across the boundary
and the validation PSNR becomes meaningless.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.common import paths  # noqa: E402
SUBMODULE = paths.use_deeptex()

from modules.utils import imresize_np  # noqa: E402


def tile_origins(extent, crop, step, thres):
    """Start offsets covering [0, extent), donor semantics (extract_subimages.py:121-126).

    The donor appends one extra flush-to-edge tile when the uncovered remainder
    exceeds thres. Unlike the donor this returns [0] instead of raising when the
    image is smaller than the crop - stage [A] has already guaranteed the whole
    image is at least gt_size, so the whole image is a legitimate single tile.
    """
    if extent <= crop:
        return [0]
    starts = list(np.arange(0, extent - crop + 1, step))
    if extent - (starts[-1] + crop) > thres:
        starts.append(extent - crop)
    return [int(s) for s in starts]


def make_lr(hr, scale, kernel):
    """Downsample an HR tile by scale, returning an 8-bit array.

    imresize_np returns float64 without rounding (its docstring says "w/o round"),
    so the quantisation to 8-bit is ours to do and must round, not truncate.
    """
    if kernel == "bicubic":
        lr = imresize_np(hr.astype(np.float64), 1.0 / scale)
        return np.rint(lr).clip(0, 255).astype(np.uint8)
    if kernel == "area":
        h, w = hr.shape[:2]
        return cv2.resize(hr, (w // scale, h // scale),
                          interpolation=cv2.INTER_AREA)
    raise ValueError("unknown kernel %s" % kernel)


def process_one(job):
    """Tile one source image. Returns (stem, [(tile_name, hr, lr), ...]) or raises."""
    path, stem, cfg = job
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError("cv2.imread returned None: %s" % path)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        # Precondition for imresize_np, which allocates with the input channel
        # count but only writes channels 0/1/2 (modules/utils.py:161,165-170).
        raise RuntimeError("%s is %s %s; stage [A] must run first"
                           % (path, arr.dtype, arr.shape))

    scale, crop, step = cfg["scale"], cfg["crop"], cfg["step"]
    thres, gt_size, kernel = cfg["thres"], cfg["gt_size"], cfg["kernel"]
    h, w = arr.shape[:2]
    out = []
    idx = 0
    for y in tile_origins(h, crop, step, thres):
        for x in tile_origins(w, crop, step, thres):
            hr = arr[y:y + min(crop, h), x:x + min(crop, w)]
            # Trim to a multiple of scale so the x4 relation is exact even for the
            # whole-image single-tile case.
            th, tw = hr.shape[0] - hr.shape[0] % scale, hr.shape[1] - hr.shape[1] % scale
            hr = np.ascontiguousarray(hr[:th, :tw])
            if min(th, tw) < gt_size:
                continue
            lr = make_lr(hr, scale, kernel)

            assert hr.dtype == np.uint8 and lr.dtype == np.uint8, (hr.dtype, lr.dtype)
            assert hr.ndim == 3 and hr.shape[2] == 3, hr.shape
            assert lr.ndim == 3 and lr.shape[2] == 3, lr.shape
            assert hr.shape[0] == lr.shape[0] * scale, (hr.shape, lr.shape)
            assert hr.shape[1] == lr.shape[1] * scale, (hr.shape, lr.shape)
            assert min(hr.shape[:2]) >= gt_size, hr.shape
            assert min(lr.shape[:2]) >= gt_size // scale, lr.shape
            assert np.isfinite(hr).all() and np.isfinite(lr).all()

            idx += 1
            out.append(("%s_s%03d.png" % (stem, idx), hr, lr))
    return stem, out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="normalised image dir (stage [A] output)")
    ap.add_argument("--out", dest="dst", required=True, help="dataset dir to create HR/ LRx4/ under")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--gt-size", type=int, default=128, help="must match cfg gt_size")
    ap.add_argument("--crop-size", type=int, default=480, help="donor default")
    ap.add_argument("--step", type=int, default=240, help="donor default")
    ap.add_argument("--thres-size", type=int, default=48, help="donor default")
    ap.add_argument("--lr-kernel", default="bicubic", choices=["bicubic", "area"],
                    help="bicubic = imresize_np (what upstream trains on); "
                         "area = cv2.INTER_AREA (what matsynth_extract.py used for MatUp's LR)")
    ap.add_argument("--holdout", type=float, default=0.1,
                    help="fraction of SOURCE IMAGES held out for validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=0, help="0 = min(8, cpu_count)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for n in ("crop_size", "step", "thres_size"):
        v = getattr(args, n)
        if v % args.scale:
            raise SystemExit("--%s must be a multiple of --scale" % n.replace("_", "-"))
    if args.crop_size < args.gt_size:
        raise SystemExit("--crop-size must be >= --gt-size")

    if not os.path.isdir(args.src):
        raise SystemExit("no such directory: %s" % args.src)
    srcs = sorted(f for f in os.listdir(args.src) if f.lower().endswith(".png"))
    if not srcs:
        raise SystemExit("no PNGs under %s (did stage [A] run?)" % args.src)

    dirs = {k: os.path.join(args.dst, k) for k in ("HR", "LRx4", "HR_val", "LRx4_val")}
    for d in dirs.values():
        if os.path.isdir(d) and os.listdir(d) and not args.force:
            raise SystemExit("%s is not empty (use --force)" % d)
        os.makedirs(d, exist_ok=True)

    stems = [os.path.splitext(f)[0] for f in srcs]
    rng = random.Random(args.seed)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * args.holdout))
    val_stems = set(shuffled[:n_val])
    print("%d source images, %d held out for validation (seed %d)"
          % (len(stems), len(val_stems), args.seed))

    cfg = {"scale": args.scale, "crop": args.crop_size, "step": args.step,
           "thres": args.thres_size, "gt_size": args.gt_size, "kernel": args.lr_kernel}
    jobs = [(os.path.join(args.src, f), os.path.splitext(f)[0], cfg) for f in srcs]

    n_jobs = args.jobs or min(8, mp.cpu_count() or 1)
    tile_map, n_train, n_val_tiles = {}, 0, 0

    def emit(stem, tiles):
        nonlocal n_train, n_val_tiles
        is_val = stem in val_stems
        hr_dir = dirs["HR_val"] if is_val else dirs["HR"]
        lr_dir = dirs["LRx4_val"] if is_val else dirs["LRx4"]
        for name, hr, lr in tiles:
            cv2.imwrite(os.path.join(hr_dir, name), hr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            cv2.imwrite(os.path.join(lr_dir, name), lr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        tile_map[stem] = {"split": "val" if is_val else "train",
                          "tiles": [t[0] for t in tiles]}
        if is_val:
            n_val_tiles += len(tiles)
        else:
            n_train += len(tiles)

    if n_jobs > 1:
        # imap_unordered re-raises in the parent; the donor's apply_async without
        # error_callback swallows worker exceptions entirely.
        with mp.Pool(n_jobs) as pool:
            for i, (stem, tiles) in enumerate(pool.imap_unordered(process_one, jobs), 1):
                emit(stem, tiles)
                if i % 25 == 0 or i == len(jobs):
                    print("  %d/%d sources, %d train + %d val tiles"
                          % (i, len(jobs), n_train, n_val_tiles))
    else:
        for i, job in enumerate(jobs, 1):
            stem, tiles = process_one(job)
            emit(stem, tiles)
            if i % 25 == 0 or i == len(jobs):
                print("  %d/%d sources, %d train + %d val tiles"
                      % (i, len(jobs), n_train, n_val_tiles))

    empty = sorted(s for s, v in tile_map.items() if not v["tiles"])
    manifest = {
        "source": os.path.abspath(args.src),
        "output": os.path.abspath(args.dst),
        "args": {"scale": args.scale, "gt_size": args.gt_size,
                 "crop_size": args.crop_size, "step": args.step,
                 "thres_size": args.thres_size, "lr_kernel": args.lr_kernel,
                 "holdout": args.holdout, "seed": args.seed},
        "counts": {"sources": len(srcs), "val_sources": len(val_stems),
                   "train_tiles": n_train, "val_tiles": n_val_tiles,
                   "sources_with_no_tiles": len(empty)},
        "sources_with_no_tiles": empty,
        "tiles_by_source": tile_map,
    }
    mpath = os.path.join(args.dst, "pairs_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    print("train %d tiles -> %s" % (n_train, dirs["HR"]))
    print("val   %d tiles -> %s" % (n_val_tiles, dirs["HR_val"]))
    if empty:
        print("[warn] %d source(s) produced no tiles: %s"
              % (len(empty), ", ".join(empty[:10])))
    print("manifest -> %s" % mpath)
    if n_train == 0:
        raise SystemExit("no training tiles produced")


if __name__ == "__main__":
    main()
