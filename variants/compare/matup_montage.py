#!/usr/bin/env python3
"""
One 4x6 contact sheet per material: LR / bilinear / MatUp / GT across all five
maps plus the render.

    docker compose run --rm dataprep python variants/compare/matup_montage.py --all

Every cell is read from what `matup_postprocess.py` produced, so the border is
already cropped to the valid region, the normal's green channel is already
flipped back to OpenGL, and all four rows are already at one size.

The bilinear row is MatUp's own initialisation (dataLoader.py resizes every map
with INTER_LINEAR x4 before the optimisation starts), so it is the row that says
whether the optimisation added detail or removed it.

Two of the six columns do not come from MatUp itself and need their own step
first, otherwise those cells are labelled instead of drawn:

* `height` - MatUp writes none. `matup/data_utils.py` writes exactly five files
  and height is not among them, and the height it reads is a fixed bilinear
  upsample that is never optimised. `scripts/matup/matup_height.py` reconstructs one
  by normal integration, which is what the paper describes in 3.3.
* `render` - MatUp renders only its own result. `scripts/matup/matup_render.py`
  renders the LR, bilinear and GT map sets through the same renderer and the
  same eval light so all four are comparable.

The DeepTextures sheet (`montage_deeptex.png`, 5x2 over baseColor and render) and
its `<material>_deeptex.png` copies in --collect are the paper arm's:
variants/deeptex/paper/deeptex_montage.py runs process() and collect() below with
its hook. Without the hook this writes montage.png, the JSONs and summary.md
exactly as the old script did when no DeepTextures column existed. build() is
unchanged and shared: its ("DeepTextures", col) placeholder note only applies
when a caller passes that row.

Nothing here is normalised for display: an all-zero metallic renders black, and
that is the honest picture. Rescaling it per-cell would make a constant map look
like a textured one.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

COLUMNS = ["baseColor", "normal", "roughness", "metallic", "height", "render"]
# label -> the filename matup_postprocess.py writes for that column
ROWS = [("LR", "lr"), ("bilinear", "bilinear"), ("MatUp", "matup"), ("GT", "gt")]

# DEEPTEX_COLUMNS / DEEPTEX_ROWS / DEEPTEX_LEFT: variants/deeptex/paper/deeptex_montage.py.

PAD = 6
LEFT = 96
HEAD = 34
TITLE = 32
FONT = cv2.FONT_HERSHEY_SIMPLEX


def read(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def to_display(arr, size):
    """16-bit or 8-bit, gray or colour -> 8-bit BGR at `size`, no rescaling."""
    if arr is None:
        return None
    a = (arr >> 8).astype(np.uint8) if arr.dtype == np.uint16 else arr.astype(np.uint8)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    interp = cv2.INTER_AREA if a.shape[0] > size else cv2.INTER_NEAREST
    return cv2.resize(a, (size, size), interpolation=interp)


def placeholder(size, note):
    cell = np.full((size, size, 3), 28, np.uint8)
    cv2.putText(cell, "n/a", (size // 2 - 26, size // 2 - 6), FONT, 0.7,
                (110, 110, 110), 2, cv2.LINE_AA)
    for i, line in enumerate(note):
        w = cv2.getTextSize(line, FONT, 0.36, 1)[0][0]
        cv2.putText(cell, line, (max(4, size // 2 - w // 2), size // 2 + 20 + i * 15),
                    FONT, 0.36, (90, 90, 90), 1, cv2.LINE_AA)
    return cell


def build(mat_dir, cmp_dir, size, columns, rows, suffix="", left=LEFT):
    """Draw one contact sheet for the given grid.

    `run_dir` used to be a parameter and was never read - process() still
    validates it, which is where that check belongs.
    """
    name = os.path.basename(mat_dir.rstrip("/"))
    stamp = os.path.basename(cmp_dir.rstrip("/"))

    # Everything now comes out of compare/, already cropped to the valid region,
    # already at one size, with the normal's green channel already flipped back
    # to OpenGL. height arrives via matup_height.py and render via
    # matup_render.py; a cell is only missing if one of those has not been run.
    cells = {}
    for col in columns:
        for row, fname in rows:
            path = os.path.join(cmp_dir, col, fname + ".png")
            cells[(row, col)] = read(path) if os.path.exists(path) else None

    notes = {("MatUp", "height"): ["run matup_height.py"]}
    for r, _ in rows:
        notes.setdefault((r, "render"), ["run matup_render.py"])
    for col in columns:
        # Overwrite the render note too: "run matup_render.py" is the wrong advice
        # for this row - nothing MatUp does will ever fill it.
        notes[("DeepTextures", col)] = ["run", "deeptex_compare.py"]

    w = left + len(columns) * (size + PAD) + PAD
    h = TITLE + HEAD + len(rows) * (size + PAD) + PAD
    sheet = np.full((h, w, 3), 18, np.uint8)

    cv2.putText(sheet, "%s    %s%s" % (name, stamp, suffix), (10, 22), FONT, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    for c, col in enumerate(columns):
        x = left + c * (size + PAD)
        cv2.putText(sheet, col, (x + 4, TITLE + 22), FONT, 0.55,
                    (210, 210, 210), 1, cv2.LINE_AA)

    for r, (row, _) in enumerate(rows):
        y = TITLE + HEAD + r * (size + PAD)
        cv2.putText(sheet, row, (10, y + size // 2), FONT, 0.62,
                    (210, 210, 210), 1, cv2.LINE_AA)
        for c, col in enumerate(columns):
            x = left + c * (size + PAD)
            arr = cells.get((row, col))
            cell = to_display(arr, size)
            if cell is None:
                cell = placeholder(size, notes.get((row, col), []))
            sheet[y:y + size, x:x + size] = cell

    return sheet, name, stamp


def process(mat_dir, args, extra=None):
    """One material's montage.png. `extra` (paper/deeptex_montage.py's DeeptexSheet)
    may draw a second sheet from the same compare/ folder afterwards."""
    name = os.path.basename(mat_dir.rstrip("/"))
    cmps = sorted(d for d in glob.glob(os.path.join(mat_dir, "compare", "*"))
                  if os.path.isdir(d))
    if args.run:
        cmps = [d for d in cmps if os.path.basename(d) == args.run]
    if not cmps:
        print("  %-28s SKIP (run matup_postprocess.py first)" % name)
        return False
    cmp_dir = cmps[-1]
    stamp = os.path.basename(cmp_dir)

    run_dir = os.path.join(mat_dir, "matup-output-%s" % name, stamp)
    if not os.path.isdir(run_dir):
        print("  %-28s SKIP (no run folder %s)" % (name, stamp))
        return False

    sheet, _, _ = build(mat_dir, cmp_dir, args.size, COLUMNS, ROWS)
    out = os.path.join(cmp_dir, "montage.png")
    if not cv2.imwrite(out, sheet):
        raise SystemExit("cv2.imwrite failed for " + out)
    print("  %-28s %s  %dx%d" % (name, stamp, sheet.shape[1], sheet.shape[0]))

    if extra:
        extra.after_montage(mat_dir, cmp_dir, args.size, stamp)
    return True


SUMMARY_MAPS = ["baseColor", "normal", "roughness", "metallic", "height"]


def collect(mat_dirs, dest, extra=None):
    """Gather everything worth sharing into one folder.

    A single directory is the point: it is the only part of runs/ that is
    committed, so .gitignore stays two lines instead of a ladder of negations
    for metrics.json buried under compare/<timestamp>/.

    `extra` may copy one more file per material (.collect_one -> 0/1) and add
    lines above the summary table (.note_lines) and to the closing message
    (.count_note).
    """
    os.makedirs(dest, exist_ok=True)
    rows = []
    n_extra = 0
    for d in mat_dirs:
        name = os.path.basename(d.rstrip("/"))
        newest = sorted(glob.glob(os.path.join(d, "compare", "*", "montage.png")))[-1]
        cv2.imwrite(os.path.join(dest, name + ".png"),
                    cv2.imread(newest, cv2.IMREAD_UNCHANGED))

        if extra:
            n_extra += extra.collect_one(os.path.dirname(newest), dest, name)

        mpath = os.path.join(os.path.dirname(newest), "metrics.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as fh:
            m = json.load(fh)
        with open(os.path.join(dest, name + ".json"), "w") as fh:
            json.dump(m, fh, indent=2)
        rows.append((name, m))

    count_note = extra.count_note(n_extra) if extra else ""
    if not rows:
        print("collected %d montage(s)%s into %s"
              % (len(mat_dirs), count_note, dest))
        return

    lines = [
        "# MatUp vs bilinear",
        "",
        "`matup / bilinear` on mean gradient magnitude, over the valid region.",
        "bilinear is not just a baseline here - it is MatUp's own initialisation",
        "(`dataLoader.py` resizes every map INTER_LINEAR x4 before optimising), so",
        "**below 1.0 means the optimisation removed detail it started with.**",
        "",
        "`n/a` marks a constant map with no gradient to divide by (metallic is all",
        "zero on non-metals). `height` is reconstructed by scripts/matup/matup_height.py,",
        "not produced by MatUp - it adds high frequency by construction, so its",
        "column says nothing about MatUp's quality.",
        "",
        "| material | " + " | ".join(SUMMARY_MAPS) + " |",
        "|---" * (len(SUMMARY_MAPS) + 1) + "|",
    ]
    if extra:
        lines[-2:-2] = extra.note_lines(n_extra)       # just above the table header
    for name, m in sorted(rows):
        cells = []
        for k in SUMMARY_MAPS:
            v = m["maps"].get(k, {}).get("matup_over_bilinear", {}).get("grad")
            cells.append("n/a" if v is None else "%.3f" % v)
        lines.append("| %s | %s |" % (name, " | ".join(cells)))

    with open(os.path.join(dest, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("collected %d montage(s)%s + metrics + summary.md into %s"
          % (len(mat_dirs), count_note, dest))


def parser(description=__doc__):
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/runs/matsynth")
    ap.add_argument("--material")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run", help="compare/ timestamp to use; default is the newest")
    ap.add_argument("--size", type=int, default=320, help="cell size in pixels")
    ap.add_argument("--collect", help="also copy every montage into this directory")
    return ap


def run(args, extra=None):
    if args.material:
        dirs = [os.path.join(args.root, args.material)]
    elif args.all:
        dirs = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                      if os.path.isdir(os.path.join(d, "compare")))
    else:
        raise SystemExit("pass --material <name> or --all")

    print("montaging %d material(s)" % len(dirs))
    made = [d for d in dirs if process(d, args, extra)]

    if args.collect:
        collect(made, args.collect, extra)

    print("\n%d montage(s) written" % len(made))


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
