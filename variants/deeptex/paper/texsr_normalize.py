#!/usr/bin/env python3
"""Normalise an arbitrary image directory into 8-bit, 3-channel, align-multiple PNGs.

Stage [A] of the DeepTextures data pipeline. Runs in the dataprep container.

This is the step upstream deleted. The evidence that it existed is the directory
name its converter still points at: data/textures_db_normalized. What it has to
undo, measured on external/DeepTextures/data/textures_db (300 files):

    200 .png + 100 .jpg          the converter globs *.png only, so 100 are
                                 silently dropped before this script existed
    73  8-bit RGBA               alpha is discarded by decode_png(channels=3)
    69  8-bit palette            expanded by the decoder, again silently
    45  16-bit RGBA              TRUNCATED to 8-bit by decode_png's uint8 default
    13  8-bit RGB                the only ones that need nothing
    64^2 .. 2048^2               38 non-square, 46 with a side not divisible by 4,
                                 1 below gt_size

Every one of those conversions happens anyway, inside modules/dataset.py:12-13,
with no log line. Doing it here makes each one counted and reviewable, and it is a
hard precondition for stage [B]: imresize_np allocates its scratch buffer with the
INPUT channel count (modules/utils.py:161) but only ever writes channels 0/1/2
(:165-170, :190-195), so a 4-channel array comes back with its fourth channel
silently zeroed.

cv2, never PIL. scripts/compare/matup_postprocess.py already records why: PIL silently
truncates 16-bit RGB PNGs to 8-bit, which is exactly the conversion we are here to
count rather than suffer. Byte order is irrelevant because we read and write
through cv2 only (same argument as scripts/common/matsynth_extract.py:20-24).
"""

import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np

# cv2 reads all of these; the 100 .jpg files in textures_db are why the list is
# not just png.
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

OP_KEYS = ("gray_replicated", "alpha_dropped", "alpha_composited",
           "alpha_nonopaque", "u16_to_u8", "cropped_to_align")


def to_u8(arr, stats):
    """16-bit -> 8-bit by rounding, not by shifting.

    arr >> 8 biases every value low by up to 1 LSB. rint(a/257) is the exact
    inverse of the 8->16 promotion (x*257) that matup_postprocess.py's to_u16
    does, so a round trip through both is lossless.
    """
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        stats["u16_to_u8"] += 1
        return np.rint(arr / 257.0).clip(0, 255).astype(np.uint8)
    raise ValueError("unexpected dtype %s (expected uint8 or uint16)" % arr.dtype)


def fit_channels(arr, alpha_mode, gray_mode, stats):
    """Return an HxWx3 array. Counts every coercion it performs."""
    if arr.ndim == 2:
        if gray_mode == "error":
            raise ValueError("grayscale input and --gray error")
        stats["gray_replicated"] += 1
        return np.repeat(arr[:, :, None], 3, axis=2)

    nch = arr.shape[2]
    if nch == 3:
        return arr
    if nch == 1:
        if gray_mode == "error":
            raise ValueError("single-channel input and --gray error")
        stats["gray_replicated"] += 1
        return np.repeat(arr, 3, axis=2)
    if nch in (2, 4):
        # 2 = gray+alpha, 4 = BGRA
        colour = arr[:, :, :1] if nch == 2 else arr[:, :, :3]
        alpha = arr[:, :, -1]
        full = 255 if arr.dtype == np.uint8 else 65535
        if int(alpha.min()) < full:
            # Dropping alpha on a fully-opaque image is free. Dropping it on a
            # genuinely transparent one is a data decision, so count them apart.
            stats["alpha_nonopaque"] += 1
        if alpha_mode == "error":
            raise ValueError("alpha channel present and --alpha error")
        if alpha_mode == "drop":
            stats["alpha_dropped"] += 1
        else:
            bg = full if alpha_mode == "composite-white" else 0
            a = (alpha.astype(np.float64) / full)[:, :, None]
            colour = np.rint(colour.astype(np.float64) * a + bg * (1.0 - a))
            colour = colour.clip(0, full).astype(arr.dtype)
            stats["alpha_composited"] += 1
        if nch == 2:
            stats["gray_replicated"] += 1
            return np.repeat(colour, 3, axis=2)
        return colour
    raise ValueError("unsupported channel count %d" % nch)


def crop_align(arr, align, stats):
    h, w = arr.shape[:2]
    ch, cw = h - (h % align), w - (w % align)
    if (ch, cw) != (h, w):
        stats["cropped_to_align"] += 1
    return arr[:ch, :cw]


def collect(src, recursive):
    paths = []
    if recursive:
        for root, _, files in os.walk(src):
            for f in files:
                if os.path.splitext(f)[1].lower() in EXTS:
                    paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(src):
            p = os.path.join(src, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in EXTS:
                paths.append(p)
    paths.sort()
    return paths


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="source image directory")
    ap.add_argument("--out", dest="dst", required=True, help="output directory")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--align", type=int, default=4,
                    help="crop bottom/right to a multiple of this (= scale)")
    ap.add_argument("--min-size", type=int, default=128,
                    help="reject images smaller than this (= gt_size)")
    ap.add_argument("--alpha", default="drop",
                    choices=["drop", "composite-white", "composite-black", "error"])
    ap.add_argument("--gray", default="replicate", choices=["replicate", "error"])
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N (0 = all)")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit("no such source directory: %s" % args.src)

    paths = collect(args.src, args.recursive)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit("no images found under %s" % args.src)

    # Output name is <stem>.png, so 001.png and 001.jpg would collide and one
    # would silently win. In a directory that is 200 png + 100 jpg that is a real
    # risk, not a hypothetical one.
    stems = Counter(os.path.splitext(os.path.basename(p))[0] for p in paths)
    dupes = sorted(s for s, n in stems.items() if n > 1)
    if dupes:
        for s in dupes:
            same = [os.path.basename(p) for p in paths
                    if os.path.splitext(os.path.basename(p))[0] == s]
            print("[collision] %s.png <- %s" % (s, ", ".join(same)))
        raise SystemExit(
            "%d output-name collision(s); resolve them before normalising" % len(dupes))

    if os.path.isdir(args.dst) and os.listdir(args.dst) and not args.force:
        raise SystemExit("%s is not empty (use --force)" % args.dst)
    os.makedirs(args.dst, exist_ok=True)

    stats = Counter()
    records, rejected, errored = [], [], []

    for p in paths:
        name = os.path.basename(p)
        stem = os.path.splitext(name)[0]
        arr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if arr is None:
            errored.append((name, "cv2.imread returned None"))
            continue
        src_dtype, src_shape = str(arr.dtype), list(arr.shape)
        before = stats.copy()
        try:
            arr = fit_channels(arr, args.alpha, args.gray, stats)
            arr = to_u8(arr, stats)
            arr = crop_align(arr, args.align, stats)
        except ValueError as e:
            errored.append((name, str(e)))
            continue
        ops = [k for k in OP_KEYS if stats[k] != before[k]]

        if min(arr.shape[:2]) < args.min_size:
            rejected.append((name, "%dx%d < min-size %d"
                             % (arr.shape[1], arr.shape[0], args.min_size)))
            stats["rejected_too_small"] += 1
            continue

        out = os.path.join(args.dst, stem + ".png")
        # Compression level 3 matches external/esrgan-tf2/data/extract_subimages.py:22.
        if not cv2.imwrite(out, arr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            errored.append((name, "cv2.imwrite failed"))
            continue
        stats["written"] += 1
        if os.path.splitext(name)[1].lower() != ".png":
            stats["non_png_rescued"] += 1
        records.append({"src": name, "src_dtype": src_dtype, "src_shape": src_shape,
                        "ops": ops, "out": stem + ".png",
                        "out_shape": list(arr.shape)})

    # Postcondition. Stage [B] depends on every one of these (see the imresize_np
    # note in the module docstring), so assert rather than trust.
    for r in records:
        s = r["out_shape"]
        assert len(s) == 3 and s[2] == 3, r
        assert s[0] % args.align == 0 and s[1] % args.align == 0, r
        assert min(s[0], s[1]) >= args.min_size, r

    manifest = {
        "source": os.path.abspath(args.src),
        "output": os.path.abspath(args.dst),
        "args": {"align": args.align, "min_size": args.min_size,
                 "alpha": args.alpha, "gray": args.gray, "limit": args.limit},
        "counts": dict(stats),
        "found": len(paths),
        "rejected": [{"src": n, "why": w} for n, w in rejected],
        "errored": [{"src": n, "why": w} for n, w in errored],
        "files": records,
    }
    mpath = os.path.normpath(os.path.join(args.dst, os.pardir,
                                          "normalize_manifest.json"))
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    print("found %d, written %d -> %s" % (len(paths), stats["written"], args.dst))
    for k in sorted(stats):
        print("  %-22s %d" % (k, stats[k]))
    if stats["non_png_rescued"]:
        print("  (%d of those were non-PNG sources that the upstream *.png glob "
              "would have dropped)" % stats["non_png_rescued"])
    for n, w in rejected:
        print("[rejected] %s: %s" % (n, w))
    for n, w in errored:
        print("[error]    %s: %s" % (n, w))
    print("manifest -> %s" % mpath)

    if errored:
        raise SystemExit("%d image(s) failed" % len(errored))


if __name__ == "__main__":
    main()
