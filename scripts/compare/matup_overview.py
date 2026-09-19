#!/usr/bin/env python3
"""One sheet for the whole set: every material as a row, every method as a column.

    docker compose run --rm dataprep python scripts/compare/matup_overview.py \
        --map baseColor --out /workspace/runs/montages/overview_baseColor.png

matup_montage.py answers "what did the methods do to THIS material". This answers
"what do the methods do", which is the question a reader asks first and which
twelve separate files cannot show. Cells come from the same compare/ directories,
so nothing is recomputed and nothing can drift from the per-material sheets.

The gradient column is why this sheet exists rather than a table. Over the twelve
materials MatUp lands at 14-47% of the ground truth's mean gradient magnitude and
DeepTextures at 78-156%, while PSNR prefers MatUp on ten of twelve - and prefers
the untouched LR input over BOTH methods on several. Printing |grad| as a
percentage of GT next to the pictures puts that contradiction where it cannot be
missed: one method removes detail, the other invents it at roughly the right
amount, and a single number hides which.
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

COLUMNS = [("LR", "lr"), ("bilinear", "bilinear"), ("MatUp", "matup"),
           ("DeepTextures", "deeptex"), ("GT", "gt")]

PAD = 4
LEFT = 210
HEAD = 30
TITLE = 34
RIGHT = 150          # the |grad| as % of GT column
FONT = cv2.FONT_HERSHEY_SIMPLEX


def fit(text, width, start=0.5, floor=0.3):
    s = start
    while cv2.getTextSize(text, FONT, s, 1)[0][0] > width and s > floor:
        s -= 0.02
    return s


def to_display(arr, size):
    if arr is None:
        return None
    a = (arr >> 8).astype(np.uint8) if arr.dtype == np.uint16 else arr.astype(np.uint8)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    interp = cv2.INTER_AREA if a.shape[0] > size else cv2.INTER_NEAREST
    return cv2.resize(a[:, :, :3], (size, size), interpolation=interp)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/runs/matsynth")
    ap.add_argument("--map", default="baseColor")
    ap.add_argument("--size", type=int, default=190)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mats = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                  if os.path.isdir(os.path.join(d, "compare")))
    rows = []
    for d in mats:
        cmps = sorted(x for x in glob.glob(os.path.join(d, "compare", "*"))
                      if os.path.isdir(x))
        if not cmps:
            continue
        cmp_dir = cmps[-1]
        cells = {}
        for _, fname in COLUMNS:
            p = os.path.join(cmp_dir, args.map, fname + ".png")
            cells[fname] = cv2.imread(p, cv2.IMREAD_UNCHANGED) if os.path.exists(p) else None
        if cells["gt"] is None:
            continue

        # |grad| as a percentage of GT, read from the metrics the montage shares.
        pct = {}
        mpath = os.path.join(cmp_dir, "metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as fh:
                mm = json.load(fh)
            e = mm.get("maps", {}).get(args.map, {})
            g = (e.get("gt") or {}).get("grad")
            for _, fname in COLUMNS:
                v = (e.get(fname) or {}).get("grad")
                if g and v is not None:
                    pct[fname] = 100.0 * v / g
        rows.append((os.path.basename(d.rstrip("/")), cells, pct))

    if not rows:
        raise SystemExit("nothing to draw under " + args.root)

    size = args.size
    w = LEFT + len(COLUMNS) * (size + PAD) + PAD + RIGHT
    h = TITLE + HEAD + len(rows) * (size + PAD) + PAD
    sheet = np.full((h, w, 3), 18, np.uint8)

    cv2.putText(sheet, "MatUp vs DeepTextures  -  %s  -  %d materials"
                % (args.map, len(rows)), (10, 24), FONT, 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)

    for c, (lab, _) in enumerate(COLUMNS):
        x = LEFT + c * (size + PAD)
        cv2.putText(sheet, lab, (x + 3, TITLE + 20), FONT,
                    fit(lab, size - 6, 0.5), (215, 215, 215), 1, cv2.LINE_AA)
    cv2.putText(sheet, "|grad| as % of GT", (LEFT + len(COLUMNS) * (size + PAD) + 6,
                TITLE + 20), FONT, 0.42, (215, 215, 215), 1, cv2.LINE_AA)

    for r, (name, cells, pct) in enumerate(rows):
        y = TITLE + HEAD + r * (size + PAD)
        cv2.putText(sheet, name, (8, y + size // 2), FONT,
                    fit(name, LEFT - 16, 0.46), (215, 215, 215), 1, cv2.LINE_AA)
        for c, (_, fname) in enumerate(COLUMNS):
            x = LEFT + c * (size + PAD)
            cell = to_display(cells.get(fname), size)
            if cell is None:
                cell = np.full((size, size, 3), 30, np.uint8)
                cv2.putText(cell, "n/a", (size // 2 - 18, size // 2), FONT, 0.5,
                            (110, 110, 110), 1, cv2.LINE_AA)
            sheet[y:y + size, x:x + size] = cell

        xr = LEFT + len(COLUMNS) * (size + PAD) + 6
        for i, key in enumerate(("matup", "deeptex")):
            if key in pct:
                # Red when detail was lost, green when it is near or above GT.
                col = (90, 220, 90) if pct[key] >= 70 else (90, 90, 235)
                cv2.putText(sheet, "%-12s %3.0f%%" % (key, pct[key]),
                            (xr, y + size // 2 - 8 + i * 22), FONT, 0.42,
                            col, 1, cv2.LINE_AA)

    if not cv2.imwrite(args.out, sheet):
        raise SystemExit("cv2.imwrite failed for " + args.out)
    print("%d material(s) x %d column(s) -> %s  (%dx%d)"
          % (len(rows), len(COLUMNS), args.out, sheet.shape[1], sheet.shape[0]))


if __name__ == "__main__":
    main()
