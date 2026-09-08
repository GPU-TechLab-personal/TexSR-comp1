#!/usr/bin/env python3
"""
Extract one MatSynth material from a HuggingFace parquet shard into the on-disk
layout MatUp expects, plus a ground-truth copy at the super-resolved size.

Run it in the dataprep container, never in a method container:

    docker compose run --rm dataprep python scripts/matsynth_extract.py --list
    docker compose run --rm dataprep python scripts/matsynth_extract.py --index 0

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
                    help="row within the shard (see --list)")
    ap.add_argument("--out", default="/workspace/runs/matsynth",
                    help="destination root; runs/ is gitignored")
    ap.add_argument("--lr-size", type=int, default=128,
                    help="LR map size fed to MatUp (its --material_size)")
    ap.add_argument("--scale", type=int, default=4,
                    help="MatUp's fixed upscaling factor; GT is lr-size*scale")
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

    columns = ["name", "category"] + list(MAPS)
    row = None
    seen = 0
    for batch in pf.iter_batches(batch_size=1, columns=columns):
        if seen == args.index:
            row = batch
            break
        seen += 1
    if row is None:
        raise SystemExit("index %d out of range (shard has %d rows)"
                         % (args.index, pf.metadata.num_rows))

    name = row.column("name")[0].as_py()
    category = row.column("category")[0].as_py()
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)

    gt_size = args.lr_size * args.scale
    lr_dir = os.path.join(args.out, safe, "SVBRDF")
    gt_dir = os.path.join(args.out, safe, "GT")
    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    print("material : %s  (category=%s)" % (name, category))
    print("shard    : %s  row %d" % (os.path.basename(args.shard), args.index))
    print("writing  : %s" % os.path.join(args.out, safe))
    print()

    report = {}
    for column, (filename, channels) in MAPS.items():
        payload = row.column(column)[0].as_py()
        if payload is None or payload.get("bytes") is None:
            raise SystemExit("material %r has no %s map" % (name, column))

        src = decode(payload["bytes"])
        src_ch = 1 if src.ndim == 2 else src.shape[2]
        src_desc = "%dx%d %s ch=%d" % (src.shape[1], src.shape[0], src.dtype, src_ch)

        for size, folder in ((gt_size, gt_dir), (args.lr_size, lr_dir)):
            arr = fit_channels(src, channels)
            arr = resize_square(arr, size)
            write_png(os.path.join(folder, filename), as_integer(arr))

        report[column] = src_desc
        print("  %-14s %-24s -> %d^2 (LR) + %d^2 (GT)  kept %s"
              % (column, src_desc, args.lr_size, gt_size, src.dtype))

    meta = {
        "name": name,
        "category": category,
        "shard": os.path.basename(args.shard),
        "row_index": args.index,
        "lr_size": args.lr_size,
        "gt_size": gt_size,
        "source_maps": report,
        "normals": "OpenGL - pass --ogl_normals to MatUp",
        "basecolor_colour_space": "assumed sRGB; see docs/OPEN-ITEMS.md #3",
    }
    with open(os.path.join(args.out, safe, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print()
    print("done. run MatUp with explicit per-run paths (README operational note 6):")
    print()
    print("  docker compose run --rm matup python main.py \\")
    print("      --upscaler swinir --ogl_normals \\")
    print("      --input_mat       /workspace/runs/matsynth/%s/SVBRDF/ \\" % safe)
    print("      --render_dir      /workspace/runs/matsynth/%s/radiance_LR/ \\" % safe)
    print("      --upsamplings_dir /workspace/runs/matsynth/%s/radiance_Upscaled/ \\" % safe)
    print("      --output_folder   /workspace/runs/matsynth/%s/output/" % safe)


if __name__ == "__main__":
    main()
