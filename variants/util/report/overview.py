#!/usr/bin/env python3
"""One sheet for the whole set: every material as a row, every method as a column.

    docker compose run --rm dataprep python variants/util/report/overview.py \\
        --map baseColor --out /workspace/runs/montages/overview_baseColor.png

    # any other experiment: name its columns (label=file stem) and its title
    ... overview.py --root <root> --map normal --title "R0a vs R1" \\
        --column R0a=r0a --column R1=r1 --column GT=gt --out <png>

A per-material montage answers "what did the methods do to THIS material"; this
answers "what do the methods do", which is the question a reader asks first and
which twelve separate files cannot show. Cells come from the newest comparison
set of each material (variants/util/report/layout.py), so nothing is recomputed.

Pictures only, deliberately: numbers stamped beside the cells tell the eye what
to conclude before it has looked. They are in each set's metrics.json and the
collected summary.

The default columns and title are the MatUp vs DeepTextures overview of the
paper arm (runs/montages/overview_*.png).
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.util.report import layout, montage, sheets  # noqa: E402

DEFAULT_COLUMNS = [("LR", "lr"), ("bilinear", "bilinear"), ("MatUp", "matup"),
                   ("DeepTextures", "deeptex"), ("GT", "gt")]
DEFAULT_TITLE = "MatUp vs DeepTextures"


def column(text):
    label, sep, key = text.partition("=")
    if not sep or not label or not key:
        raise argparse.ArgumentTypeError("--column wants LABEL=FILE_STEM, got %r" % text)
    return label, key


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/runs/matsynth")
    ap.add_argument("--map", default="baseColor")
    ap.add_argument("--size", type=int, default=190)
    ap.add_argument("--column", type=column, action="append",
                    help="LABEL=FILE_STEM, repeatable, in order (default: %s)"
                         % " ".join("%s=%s" % c for c in DEFAULT_COLUMNS))
    ap.add_argument("--title", default=DEFAULT_TITLE)
    ap.add_argument("--require", default="gt",
                    help="leave out materials without this column (empty: keep all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    columns = args.column or DEFAULT_COLUMNS

    mats = layout.materials(args.root, layout.SET_PARENT)
    rows = montage.overview_rows(mats, args.map, columns, args.require or None)
    if not rows:
        raise SystemExit("nothing to draw under " + args.root)
    sheet = sheets.table(rows, columns, args.size, "%s  -  %s  -  %d materials"
                         % (args.title, args.map, len(rows)))
    if not cv2.imwrite(args.out, sheet):
        raise SystemExit("cv2.imwrite failed for " + args.out)
    print("%d material(s) x %d column(s) -> %s  (%dx%d)"
          % (len(rows), len(columns), args.out, sheet.shape[1], sheet.shape[0]))


if __name__ == "__main__":
    main()
