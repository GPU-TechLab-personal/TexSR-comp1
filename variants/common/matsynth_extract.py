#!/usr/bin/env python3
"""
Extract one MatSynth material from a HuggingFace parquet shard into the on-disk
layout MatUp expects, plus a ground-truth copy at the super-resolved size.

Run it in the dataprep container, never in a method container:

    docker compose run --rm dataprep python scripts/common/matsynth_extract.py --list
    docker compose run --rm dataprep python scripts/common/matsynth_extract.py --index 0
    docker compose run --rm dataprep python scripts/common/matsynth_extract.py --index 0 --count 10

Layout produced under --out:

    <name>/SVBRDF/{baseColor,height,metallic,normal,roughness}.png   128x128, the LR input
    <name>/GT/{...}.png                                             512x512, for comparison
    <name>/meta.json                                                provenance

Notes that are easy to get wrong:

* Channel order. MatUp reads with cv2.imread and then converts BGR->RGB inside
  file_utils.process_image_as_float(), for multi-channel maps only. Decoding
  here with cv2.imdecode and re-encoding with cv2.imwrite leaves the file's own
  byte order untouched, so a material extracted here is consumed exactly like
  the bundled victorian_wood example.

* Bit depth is preserved, deliberately. MatSynth height is 16-bit and uses the
  range (measured: 0-65025 on acg_wood_floor_060), and
  file_utils.process_image_as_float() has an explicit uint16 branch that divides
  by 2**16-1. Flattening height to 8-bit would therefore throw away half its
  precision for nothing. It would not stay contained either: height becomes
  displacement in the renderer, the renders are what the optimisation fits, so
  the loss would propagate into every output map. The bundled victorian_wood
  example happens to be 8-bit, which is not a constraint.

* Normals. MatSynth uses the OpenGL convention, so MatUp must be given
  --ogl_normals. MatUp's own output is hardcoded to DirectX regardless
  (README, operational note 2).

* baseColor is written as-is and MatUp applies srgb2linrgb() on load. Whether
  MatSynth basecolor really is sRGB-encoded is an open assumption; see
  docs/OPEN-ITEMS.md #3.

* Downsampling uses INTER_AREA, which area-averages. That makes the LR input a
  proper decimation of the GT rather than a point sample, so LR and GT stay a
  matched pair.
"""
import argparse
import json
import os

import cv2
import numpy as np
import pyarrow.parquet as pq

# MatSynth column -> the filename MatUp looks for (matup/dataLoader.py:36-67).
# The capital C in baseColor.png matters: the container is Linux.
MAPS = {
    "basecolor": ("baseColor.png", 3),
    "height":    ("height.png",    1),
    "metallic":  ("metallic.png",  1),
    "normal":    ("normal.png",    3),
    "roughness": ("roughness.png", 1),
}

DEFAULT_SHARD = "/data/MatSynth/data/test-00000-of-00008.parquet"


def decode(buf):
    arr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError("cv2 could not decode an image payload")
    return arr


def as_integer(arr):
    """Pass integer maps through untouched; only floats need a target depth.

    Do not be tempted to normalise everything to uint8 here - see the note on
    bit depth in the module docstring.
    """
    if arr.dtype in (np.float32, np.float64):
        return np.clip(arr * 65535.0, 0, 65535).astype(np.uint16)
    return arr


def fit_channels(arr, want):
    if want == 1:
        return arr if arr.ndim == 2 else arr[:, :, 0]
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr[:, :, :3]


def center_crop(arr, size):
    h, w = arr.shape[:2]
    if size > min(h, w):
        raise SystemExit("crop size %d exceeds source %dx%d" % (size, w, h))
    y = (h - size) // 2
    x = (w - size) // 2
    return arr[y:y + size, x:x + size]


def resize_square(arr, size):
    # INTER_AREA is the correct kernel when shrinking; it averages the block a
    # pixel covers instead of sampling one point out of it.
    interp = cv2.INTER_AREA if size < arr.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(arr, (size, size), interpolation=interp)


def write_png(path, arr):
    if not cv2.imwrite(path, arr):
        raise RuntimeError("cv2.imwrite failed for " + path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", default=DEFAULT_SHARD)
    ap.add_argument("--index", type=int, default=0,
                    help="first row within the shard (see --list)")
    ap.add_argument("--count", type=int, default=1,
                    help="how many consecutive rows to extract from --index")
    ap.add_argument("--out", default="/workspace/runs/matsynth",
                    help="destination root; runs/ is gitignored")
    ap.add_argument("--lr-size", type=int, default=128,
                    help="LR map size fed to MatUp (its --material_size)")
    ap.add_argument("--scale", type=int, default=4,
                    help="MatUp's fixed upscaling factor; GT is lr-size*scale")
    ap.add_argument("--mode", choices=["resize", "crop"], default="resize",
                    help="how the LR/GT pair is built from the 4096^2 source. "
                         "resize: decimate the whole map (GT 8x, LR 32x). "
                         "crop: take a gt-size window at NATIVE resolution as GT, "
                         "then area-average that window x4 for LR.")
    ap.add_argument("--list", action="store_true",
                    help="list the materials in the shard and exit")
    args = ap.parse_args()

    pf = pq.ParquetFile(args.shard)

    if args.list:
        table = pf.read(columns=["name", "category"])
        names = table.column("name").to_pylist()
        cats = table.column("category").to_pylist()
        print("%d rows in %s" % (pf.metadata.num_rows, os.path.basename(args.shard)))
        for i, (n, c) in enumerate(zip(names, cats)):
            print("  [%2d] %-40s category=%s" % (i, n, c))
        return

    last = args.index + args.count - 1
    if last >= pf.metadata.num_rows:
        raise SystemExit("rows %d-%d requested but shard has %d"
                         % (args.index, last, pf.metadata.num_rows))

    columns = ["name", "category"] + list(MAPS)
    extracted = []
    # One pass over the shard: iter_batches streams, so asking for row 9 by
    # re-opening the file nine times would re-read everything nine times.
    for position, row in enumerate(pf.iter_batches(batch_size=1, columns=columns)):
        if position < args.index:
            continue
        if position > last:
            break
        extracted.append(extract_row(row, position, args, gt_size=args.lr_size * args.scale))

    print()
    print("extracted %d material(s) into %s" % (len(extracted), args.out))
    for safe in extracted:
        print("  " + safe)
    print()
    print("run MatUp per material with explicit per-run paths")
    print("(README operational note 6 - shared default paths destroy each other):")
    print()
    root = args.out.rstrip("/")
    print('  $m = "%s"' % extracted[0])
    print("  docker compose run --rm matup python main.py --upscaler swinir --ogl_normals `")
    print('      --input_mat       "%s/$m/SVBRDF/" `' % root)
    print('      --render_dir      "%s/$m/radiance_LR/" `' % root)
    print('      --upsamplings_dir "%s/$m/radiance_Upscaled/" `' % root)
    print('      --output_folder   "%s/$m/matup-output-$m/"' % root)


def extract_row(row, position, args, gt_size):
    name = row.column("name")[0].as_py()
    category = row.column("category")[0].as_py()
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)

    lr_dir = os.path.join(args.out, safe, "SVBRDF")
    gt_dir = os.path.join(args.out, safe, "GT")
    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    print("[%2d] %s  (category=%s)" % (position, name, category))

    report = {}
    for column, (filename, channels) in MAPS.items():
        payload = row.column(column)[0].as_py()
        if payload is None or payload.get("bytes") is None:
            raise SystemExit("material %r has no %s map" % (name, column))

        src = decode(payload["bytes"])
        src_ch = 1 if src.ndim == 2 else src.shape[2]
        src_desc = "%dx%d %s ch=%d" % (src.shape[1], src.shape[0], src.dtype, src_ch)

        arr = fit_channels(src, channels)
        if args.mode == "crop":
            # GT is a native-resolution window; LR is that same window decimated
            # x4. Decimating the WHOLE 4096^2 map instead (mode=resize) band-limits
            # the 128^2 input so hard that there is no pixel-level structure left
            # for the RGB upsampler to enhance, and MatUp then converges to
            # something smoother than its own bilinear initialisation.
            gt_arr = center_crop(arr, gt_size)
            lr_arr = resize_square(gt_arr, args.lr_size)
        else:
            gt_arr = resize_square(arr, gt_size)
            lr_arr = resize_square(arr, args.lr_size)
        write_png(os.path.join(gt_dir, filename), as_integer(gt_arr))
        write_png(os.path.join(lr_dir, filename), as_integer(lr_arr))

        report[column] = src_desc

    print("     " + "  ".join("%s %s" % (c, d.split()[1]) for c, d in report.items()))

    meta = {
        "name": name,
        "category": category,
        "shard": os.path.basename(args.shard),
        "row_index": position,
        "mode": args.mode,
        "lr_size": args.lr_size,
        "gt_size": gt_size,
        "source_maps": report,
        "normals": "OpenGL - pass --ogl_normals to MatUp",
        "basecolor_colour_space": "assumed sRGB; see docs/OPEN-ITEMS.md #3",
    }
    with open(os.path.join(args.out, safe, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    return safe


if __name__ == "__main__":
    main()
