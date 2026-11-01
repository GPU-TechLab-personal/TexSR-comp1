#!/usr/bin/env python3
"""
Turn each MatUp run into a comparison set: LR / bilinear / MatUp / GT columns per
map and for the render, cropped to the valid region at one size.

Run it in the dataprep container:

    docker compose run --rm dataprep python variants/matup/baseline/postprocess.py --all --metrics

Writes <root>/<m>/compare/<stamp>/ (the layout of variants/util/report/layout.py):
<map>/<column>.png, sheet_<map>.png, render/<column>.png, sheet_render.png and,
with --metrics, metrics.json. How the MatUp output is normalised (border, normal
flip, size) is matup_outputs.py.

A `bilinear` column is produced alongside because it is *also* MatUp's own
initialisation (`dataLoader.py`, cv2.INTER_LINEAR x4). Comparing against it is
what tells you whether the optimisation added detail or removed it.

`height` and `render` only appear once `matup_height.py` and `matup_render.py`
have run - MatUp writes neither.

Another method adds its own column through the `extra` hook of process()
(variants/deeptex/paper/deeptex_postprocess.py is the DeepTextures one).
"""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.matup.baseline import matup_outputs as mo  # noqa: E402
from variants.util.report import layout, sheets  # noqa: E402
from variants.util.report.images import read, to_float, to_u16, write  # noqa: E402
from variants.util.report.metrics import corr, full_reference, ratio, sharpness  # noqa: E402


def process(mat_dir, args, extra=None):
    """One material. `extra` adds one more column (paper/deeptex_postprocess.py's
    DeeptexColumn): .order, .add_map, .add_render, .add_metrics, .ratio_note,
    .psnr_note. Without it the four MatUp columns are written."""
    name = os.path.basename(mat_dir.rstrip("/"))
    run_dir = mo.newest_run(mat_dir, name, args.run)
    if not run_dir:
        print("  %-28s SKIP (no MatUp output)" % name)
        return None
    stamp = os.path.basename(run_dir)
    column_order = extra.order if extra else mo.COLUMNS

    lrs, gts, opts = {}, {}, {}
    for m, (src_name, opt_name) in mo.MAPS.items():
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
    measured, sides = mo.detect_border(
        [a for m, a in opts.items() if m != "normal" and m not in mo.PRECROPPED])
    border = measured
    if measured != nominal:
        print("  ! %s: border measured %d (sides %s) but tile_width %d implies %d"
              % (name, measured, sides, args.tile_width, nominal))
        print("    using the measured value")
    scale = gts["baseColor"].shape[0] // lrs["baseColor"].shape[0]
    if border % scale:
        print("  ! %s: border %d is not divisible by scale %d - LR column will be "
              "misaligned by up to %d LR pixels" % (name, border, scale, border % scale))

    out_root = layout.set_dir(mat_dir, stamp)
    report = {"material": name, "run": stamp, "border": border,
              "scale": scale, "size": gts["baseColor"].shape[0] - 2 * border,
              "maps": {}}

    for m in opts:
        cols = mo.build_columns(lrs[m], gts[m], opts[m], border, scale,
                                m == "normal", m in mo.PRECROPPED)
        if extra:
            extra.add_map(cols, name, m, border, report["size"])
        order = [c for c in column_order if c in cols]
        for col in order:
            write(layout.cell_path(out_root, m, col), cols[col])
        if not args.no_sheet:
            write(os.path.join(out_root, "sheet_%s.png" % m),
                  sheets.strip(cols, "%s  -  %s  (%s)" % (name, m, stamp), order))

        if args.metrics:
            gb, lb = sharpness(cols["bilinear"])
            go, lo = sharpness(cols["matup"])
            gg, lg = sharpness(cols["gt"])
            entry = {
                "bilinear": {"grad": gb, "lapvar": lb},
                "matup":    {"grad": go, "lapvar": lo},
                "gt":       {"grad": gg, "lapvar": lg},
                "matup_over_bilinear": {"grad": ratio(go, gb), "lapvar": ratio(lo, lb)},
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
            # Added after the sharpness keys; the committed runs/montages/*.json
            # carry the same keys in the same order.
            entry["vs_gt"] = full_reference(cols)
            report["maps"][m] = entry

    # renders: no bilinear upsample - a render is not an upsample
    side = report["size"]
    rcols = {}
    for col, fname in mo.RENDER.items():
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            continue                    # matup_render.py has not run yet
        a = read(path)
        if col not in mo.PRECROPPED_RENDER and a.shape[0] > side:
            a = a[border:-border, border:-border] if border else a
        if a.shape[0] != side:          # the LR render is native 128^2
            a = cv2.resize(a, (side, side), interpolation=cv2.INTER_NEAREST)
        rcols[col] = to_u16(a)
    if extra:
        extra.add_render(rcols, name, border, side)
    if rcols:
        for col, a in rcols.items():
            write(layout.cell_path(out_root, "render", col), a)
        if not args.no_sheet:
            ordered = {k: rcols[k] for k in column_order if k in rcols}
            write(os.path.join(out_root, "sheet_render.png"),
                  sheets.strip(ordered, "%s  -  render  (%s)" % (name, stamp),
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
        dirs = layout.materials(args.root, "SVBRDF")
    else:
        raise SystemExit("pass --material <name> or --all")

    print("postprocessing %d material(s)" % len(dirs))
    done = [r for r in (process(d, args, extra) for d in dirs) if r]
    print("\n%d material(s) written" % len(done))


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
