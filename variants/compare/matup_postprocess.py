#!/usr/bin/env python3
"""
Turn a MatUp run into a comparison-ready LR / bilinear / MatUp / GT set.

Run it in the dataprep container:

    docker compose run --rm dataprep python variants/compare/matup_postprocess.py --all --metrics

Three things have to be normalised before MatUp output can be put next to the
ground truth, or differences that have nothing to do with quality get read as
quality differences:

* **The 8px black border.** The paper optimises only over valid patches ("we use
  a sliding window mechanism over valid patches to avoid the borders"), so the
  rim is absent prediction padded with zeros, not data. Its width is
  `tile_width // 2`, not a constant 8 - change --tile_width and it moves.

* **The normal map's green channel.** `matup/data_utils.py` calls `write_normal`
  with a hardcoded `opengl_normals=False`, so the output is always in the
  internal (DirectX, +Y down) convention no matter how the input was read.
  MatSynth is OpenGL ("We adopt the OpenGL standard for the Normal map"), so the
  two disagree and bumps read as dents. Measured Y correlation against GT:
  -0.986 on the brick, -0.809 on the foam; flipping G makes both positive.

* **Size.** LR is 128^2, everything else 512^2.

A `bilinear` column is produced alongside because it is *also* MatUp's own
initialisation (`dataLoader.py`, cv2.INTER_LINEAR x4). Comparing against it is
what tells you whether the optimisation added detail or removed it.

`height` and `render` only appear once `matup_height.py` and `matup_render.py`
have run - MatUp writes neither. Only cv2 and numpy are used - PIL silently truncates 16-bit RGB PNGs to
8-bit, which is enough to invalidate every measurement below.

This is the part both methods share. The old `--deeptex-root` (a fifth column, on
baseColor and render only, for the paper arm) is variants/deeptex/paper/
deeptex_postprocess.py, which runs process() below with an `extra` column hook;
without one, process() writes exactly what the old script wrote without
--deeptex-root. deeptex_column() stays here: the MatSynth and render-loss reports
crop their DeepTextures outputs with it too.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

# texsr_metrics lives in variants/util because BOTH methods must measure with
# the same code - see the note in that file about runs/montages/ being committed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 2)))
from variants.util import texsr_metrics  # noqa: E402
from variants.compare.matup_runs import newest_run  # noqa: E402

# map name -> (LR/GT filename, MatUp output filename)
MAPS = {
    "baseColor": ("baseColor.png", "baseColor_opt.png"),
    "normal":    ("normal.png",    "normal_opt.png"),
    "roughness": ("roughness.png", "roughness_opt.png"),
    "metallic":  ("metallic.png",  "metallic_opt.png"),
    # matup_height.py reconstructs this (paper 3.3); MatUp itself writes none.
    # It comes out already cropped to the valid region.
    "height":    ("height.png",    "height_opt.png"),
}
PRECROPPED = {"height"}

# matup_render.py renders all of these with MatUp's own renderer and eval light.
# The bilinear one is MatUp's own initialisation rendered, which is what makes it
# the baseline the paper compares against (Fig. 8).
RENDER = {"lr": "render_lr.png", "bilinear": "render_bilinear.png",
          "matup": "render_opt.png", "gt": "render_gt.png"}
PRECROPPED_RENDER = {"gt", "bilinear"}   # both are produced at 496^2 already

COLUMNS = ["lr", "bilinear", "matup", "gt"]

# The fifth (DeepTextures) column and its order: variants/deeptex/paper/
# deeptex_postprocess.py (DEEPTEX_COLUMN, DEEPTEX_MAPS, ORDER_WITH_DEEPTEX).
U16_MAX = 65535


# --------------------------------------------------------------------------- io

def read(path):
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise SystemExit("cannot read " + path)
    return arr


def to_u16(arr):
    """Promote to 16-bit so every column shares a dtype. 257 = 65535/255."""
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        return arr.astype(np.uint16) * 257
    raise SystemExit("unexpected dtype %s" % arr.dtype)


def to_float(arr):
    return arr.astype(np.float64) / (U16_MAX if arr.dtype == np.uint16 else 255.0)


def write(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, arr):
        # cv2.imwrite returns False instead of raising - a missing directory or a
        # bad path fails silently otherwise.
        raise SystemExit("cv2.imwrite failed for " + path)


# ---------------------------------------------------------------- border detect

def detect_border(opt_arrays):
    """Count the zero rim MatUp padded on, from the images themselves.

    A row counts as border only when it is zero in *every* map passed in. Some
    materials have an all-zero metallic (the brick does), so testing one map
    alone would swallow the whole image.

    Callers must NOT pass the normal map. It is written through `write_normal`,
    which encodes as `65535 * (0.5 * n + 0.5)`, so its zero-padded rim comes out
    at 32767 rather than 0 and would hide the border entirely.
    """
    peak = None
    for arr in opt_arrays:
        m = arr.max(axis=2) if arr.ndim == 3 else arr
        peak = m if peak is None else np.maximum(peak, m)

    h, w = peak.shape
    top = bottom = left = right = 0
    while top < h and peak[top].max() == 0:
        top += 1
    while bottom < h and peak[h - 1 - bottom].max() == 0:
        bottom += 1
    while left < w and peak[:, left].max() == 0:
        left += 1
    while right < w and peak[:, w - 1 - right].max() == 0:
        right += 1
    return min(top, bottom, left, right), (top, bottom, left, right)


# ------------------------------------------------------------------- processing

def build_columns(lr, gt, opt, border, scale, is_normal, opt_precropped=False):
    """Crop all four views to the same valid region at the same size."""
    hr = gt.shape[0]
    b = border
    lb = b // scale

    # a[b:-b] degenerates to an empty slice when b == 0, so guard it rather than
    # writing empty PNGs (cv2.imwrite asserts instead of returning False).
    crop_hr = (lambda a: a[b:-b, b:-b]) if b else (lambda a: a)

    # bilinear must upsample the WHOLE LR map and only then crop, so that it is
    # bit-for-bit the operation MatUp initialises from.
    bilinear = crop_hr(cv2.resize(lr, (hr, hr), interpolation=cv2.INTER_LINEAR))

    # nearest for the LR column on purpose: bilinear here would make it a second
    # bilinear result rather than a view of the actual input.
    lr_view = lr[lb:-lb, lb:-lb] if lb else lr
    side = (hr - 2 * b)
    lr_view = cv2.resize(lr_view, (side, side), interpolation=cv2.INTER_NEAREST)

    matup = (opt if opt_precropped else crop_hr(opt)).copy()
    if is_normal:
        m = to_u16(matup)
        m[..., 1] = U16_MAX - m[..., 1]      # DirectX -> OpenGL, value inversion
        matup = m

    return {
        "lr":       to_u16(lr_view),
        "bilinear": to_u16(bilinear),
        "matup":    to_u16(matup),
        "gt":       to_u16(crop_hr(gt)),
    }


def make_sheet(cols, title, order=None):
    """Contact sheet for eyeballing. 8-bit: cv2.putText will not draw on 16-bit."""
    tiles = []
    for name in (order or COLUMNS):
        a = cols[name]
        a8 = (a >> 8).astype(np.uint8)
        if a8.ndim == 2:
            a8 = cv2.cvtColor(a8, cv2.COLOR_GRAY2BGR)
        bar = np.zeros((30, a8.shape[1], 3), np.uint8)
        cv2.putText(bar, name, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, a8]))
        tiles.append(np.full((tiles[-1].shape[0], 4, 3), 255, np.uint8))
    sheet = np.hstack(tiles[:-1])
    head = np.zeros((26, sheet.shape[1], 3), np.uint8)
    cv2.putText(head, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, sheet])


# ---------------------------------------------------------------------- metrics

def sharpness(arr):
    g = to_float(arr)
    g = g.mean(axis=2) if g.ndim == 3 else g
    gx, gy = np.gradient(g)
    return float(np.hypot(gx, gy).mean()), float(cv2.Laplacian(g, cv2.CV_64F).var())


def corr(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------------- per-material

def deeptex_column(path, border, side):
    """Crop a DeepTextures SR output into the same valid region as the MatUp column.

    MatUp's 512^2 contains only 496^2 of prediction; the 8px rim is zero padding it
    never optimised (README operational note 3). DeepTextures has no such rim - its
    whole output is valid - so cropping it is not symmetry for its own sake, it is
    the only way both columns cover the same pixels of the same material. Not doing
    it would measure MatUp on padding.
    """
    a = read(path)
    if a.shape[0] > side and border:
        a = a[border:-border, border:-border]
    if a.shape[0] != side:
        a = cv2.resize(a, (side, side), interpolation=cv2.INTER_NEAREST)
    return to_u16(a)


def full_reference(cols, gt_key="gt"):
    """PSNR/SSIM of every non-GT column against GT.

    New in this pass. Until DeepTextures existed there was no second method to
    compare against GT in the same units, so the chain measured only reference-free
    sharpness. Both sides are quantised to 8 bits first (texsr_metrics.full_reference)
    because MatUp writes 16-bit and DeepTextures writes 8-bit, and scoring at 16
    would hand MatUp precision the other method structurally cannot have.
    """
    out = {}
    for col, arr in cols.items():
        if col == gt_key:
            continue
        try:
            out[col] = texsr_metrics.full_reference(arr, cols[gt_key])
        except ValueError as e:
            out[col] = {"error": str(e)}
    return out




def process(mat_dir, args, extra=None):
    """One material. `extra` adds one more column (paper/deeptex_postprocess.py's
    DeeptexColumn): .order, .add_map, .add_render, .add_metrics, .ratio_note,
    .psnr_note. Without it the four MatUp columns are written as they always were.
    """
    name = os.path.basename(mat_dir.rstrip("/"))
    run_dir = newest_run(mat_dir, name, args.run)
    if not run_dir:
        print("  %-28s SKIP (no MatUp output)" % name)
        return None
    stamp = os.path.basename(run_dir)
    column_order = extra.order if extra else COLUMNS

    lrs, gts, opts = {}, {}, {}
    for m, (src_name, opt_name) in MAPS.items():
        opt_path = os.path.join(run_dir, opt_name)
        if not os.path.exists(opt_path):
            # height only exists once matup_height.py has run; everything else
            # is written by MatUp itself and its absence is a real problem.
            if m == "height":
                continue
            raise SystemExit("missing " + opt_path)
        lrs[m] = read(os.path.join(mat_dir, "SVBRDF", src_name))
        gts[m] = read(os.path.join(mat_dir, "GT", src_name))
        opts[m] = read(opt_path)

    nominal = args.tile_width // 2
    # normal encodes its zero rim as mid-grey, and the pre-cropped maps have no
    # rim left to find - neither can take part in detecting the border.
    measured, sides = detect_border(
        [a for m, a in opts.items() if m != "normal" and m not in PRECROPPED])
    border = measured
    if measured != nominal:
        print("  ! %s: border measured %d (sides %s) but tile_width %d implies %d"
              % (name, measured, sides, args.tile_width, nominal))
        print("    using the measured value")
    scale = gts["baseColor"].shape[0] // lrs["baseColor"].shape[0]
    if border % scale:
        print("  ! %s: border %d is not divisible by scale %d - LR column will be "
              "misaligned by up to %d LR pixels" % (name, border, scale, border % scale))

    out_root = os.path.join(mat_dir, "compare", stamp)
    report = {"material": name, "run": stamp, "border": border,
              "scale": scale, "size": gts["baseColor"].shape[0] - 2 * border,
              "maps": {}}

    for m in opts:
        cols = build_columns(lrs[m], gts[m], opts[m], border, scale,
                             m == "normal", m in PRECROPPED)
        if extra:
            extra.add_map(cols, name, m, border, report["size"])
        order = [c for c in column_order if c in cols]
        for col in order:
            write(os.path.join(out_root, m, col + ".png"), cols[col])
        if not args.no_sheet:
            write(os.path.join(out_root, "sheet_%s.png" % m),
                  make_sheet(cols, "%s  -  %s  (%s)" % (name, m, stamp), order))

        if args.metrics:
            gb, lb = sharpness(cols["bilinear"])
            go, lo = sharpness(cols["matup"])
            gg, lg = sharpness(cols["gt"])
            entry = {
                "bilinear": {"grad": gb, "lapvar": lb},
                "matup":    {"grad": go, "lapvar": lo},
                "gt":       {"grad": gg, "lapvar": lg},
                "matup_over_bilinear": {"grad": go / gb if gb else None,
                                        "lapvar": lo / lb if lb else None},
            }
            if m == "normal":
                # The matup column is already flipped; recover the raw one to show
                # what the correction did.
                flipped = to_float(cols["matup"])[..., 1]
                entry["normal_Y_corr_vs_gt"] = {
                    "before_flip": corr(1.0 - flipped, to_float(cols["gt"])[..., 1]),
                    "after_flip":  corr(flipped,       to_float(cols["gt"])[..., 1]),
                }
            if extra:
                extra.add_metrics(entry, cols, gb, lb)
            # Added keys only; every key above keeps the value it had before
            # texsr_metrics existed, so the committed runs/montages/*.json stay
            # comparable.
            entry["vs_gt"] = full_reference(cols)
            report["maps"][m] = entry

    # renders: three columns, no bilinear - a render is not an upsample
    side = report["size"]
    rcols = {}
    for col, fname in RENDER.items():
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            continue                    # matup_render.py has not run yet
        a = read(path)
        if col not in PRECROPPED_RENDER and a.shape[0] > side:
            a = a[border:-border, border:-border] if border else a
        if a.shape[0] != side:          # the LR render is native 128^2
            a = cv2.resize(a, (side, side), interpolation=cv2.INTER_NEAREST)
        rcols[col] = to_u16(a)
    if extra:
        extra.add_render(rcols, name, border, side)
    if rcols:
        for col, a in rcols.items():
            write(os.path.join(out_root, "render", col + ".png"), a)
        if not args.no_sheet:
            ordered = {k: rcols[k] for k in column_order if k in rcols}
            write(os.path.join(out_root, "sheet_render.png"),
                  make_sheet(ordered, "%s  -  render  (%s)" % (name, stamp),
                             list(ordered)))
        report["render_columns"] = sorted(rcols)
        if args.metrics and "gt" in rcols:
            rentry = {c: dict(zip(("grad", "lapvar"), sharpness(a)))
                      for c, a in rcols.items()}
            rentry["vs_gt"] = full_reference(rcols)
            report["render_metrics"] = rentry

    if args.metrics:
        with open(os.path.join(out_root, "metrics.json"), "w") as fh:
            json.dump(report, fh, indent=2)

    print("  %-28s %s  border=%d  -> %d^2   maps=%d  render=%d"
          % (name, stamp, border, report["size"], len(opts), len(rcols)))
    if args.metrics:
        for m in opts:
            r = report["maps"][m]["matup_over_bilinear"]
            # A constant map (metallic is all zero on non-metals) has no gradient
            # to take a ratio against; say so instead of printing a fake number.
            fmt = lambda v: "  n/a  " if v is None else "%.3fx" % v
            line = "      %-10s matup/bilinear  |grad| %s  Lapvar %s" % (
                m, fmt(r["grad"]), fmt(r["lapvar"]))
            if extra:
                line += extra.ratio_note(report["maps"][m], fmt)
            v = report["maps"][m].get("vs_gt", {})
            pf = lambda e: ("identical" if e.get("identical")
                            else ("n/a" if e.get("psnr") is None else "%.2f" % e["psnr"]))
            if "matup" in v:
                line += "   PSNR matup %s" % pf(v["matup"])
                if extra:
                    line += extra.psnr_note(v, pf)
            if m == "normal":
                c = report["maps"][m]["normal_Y_corr_vs_gt"]
                line += "   Y corr %+.3f -> %+.3f" % (c["before_flip"], c["after_flip"])
            print(line)
    return report


def parser(description=__doc__):
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/runs/matsynth")
    ap.add_argument("--material")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run", help="output timestamp to use; default is the newest")
    ap.add_argument("--tile-width", type=int, default=17,
                    help="MatUp's --tile_width; the border should be half of it")
    ap.add_argument("--no-sheet", action="store_true")
    ap.add_argument("--metrics", action="store_true")
    return ap


def run(args, extra=None):
    if args.material:
        dirs = [os.path.join(args.root, args.material)]
    elif args.all:
        dirs = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                      if os.path.isdir(os.path.join(d, "SVBRDF")))
    else:
        raise SystemExit("pass --material <name> or --all")

    print("postprocessing %d material(s)" % len(dirs))
    done = [r for r in (process(d, args, extra) for d in dirs) if r]
    print("\n%d material(s) written" % len(done))


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
