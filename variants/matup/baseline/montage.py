#!/usr/bin/env python3
"""
One 4x6 contact sheet per material: LR / bilinear / MatUp / GT across all five
maps plus the render, and --collect into the shared runs/montages/.

    docker compose run --rm dataprep python variants/matup/baseline/montage.py --all

Every cell is read from the comparison set postprocess.py wrote
(<m>/compare/<stamp>/), so the border is already cropped to the valid region,
the normal's green channel is already flipped back to OpenGL, and all four rows
are already at one size. Drawing is variants/util/report (montage.montage,
sheets.grid).

The bilinear row is MatUp's own initialisation (dataLoader.py resizes every map
with INTER_LINEAR x4 before the optimisation starts), so it is the row that says
whether the optimisation added detail or removed it.

Two of the six columns do not come from MatUp itself and need their own step
first, otherwise those cells are labelled instead of drawn:

* `height` - MatUp writes none (`matup/data_utils.py` writes exactly five files,
  and the height it reads is a fixed bilinear upsample that is never optimised).
  `matup_height.py` reconstructs one by normal integration, as the paper's 3.3
  describes.
* `render` - MatUp renders only its own result. `matup_render.py` renders the LR,
  bilinear and GT map sets through the same renderer and the same eval light so
  all four are comparable.

Another method adds a sheet or collected file through the `extra` hook of
process() / collect() (variants/deeptex/paper/deeptex_montage.py adds
montage_deeptex.png).

Nothing here is normalised for display: an all-zero metallic renders black, and
that is the honest picture.
"""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.util.report import layout, montage, sheets  # noqa: E402

COLUMNS = ["baseColor", "normal", "roughness", "metallic", "height", "render"]
# label -> the file stem postprocess.py writes for that column
ROWS = [("LR", "lr"), ("bilinear", "bilinear"), ("MatUp", "matup"), ("GT", "gt")]


def notes_for(rows, columns):
    """What would fill an empty cell, per (row label, map)."""
    notes = {("MatUp", "height"): ["run matup_height.py"]}
    for r, _ in rows:
        notes.setdefault((r, "render"), ["run matup_render.py"])
    for col in columns:
        # Overwrite the render note too: "run matup_render.py" is the wrong advice
        # for this row - nothing MatUp does will ever fill it.
        notes[("DeepTextures", col)] = ["run", "deeptex_compare.py"]
    return notes


def build(mat_dir, cmp_dir, size, columns, rows, suffix="", left=sheets.GRID.left):
    """One contact sheet of the comparison set `cmp_dir` -> (sheet, name, stamp)."""
    name = os.path.basename(mat_dir.rstrip("/"))
    stamp = os.path.basename(cmp_dir.rstrip("/"))
    sheet = montage.montage(cmp_dir, columns, rows, size,
                            "%s    %s%s" % (name, stamp, suffix),
                            notes_for(rows, columns), left)
    return sheet, name, stamp


def process(mat_dir, args, extra=None):
    """One material's montage.png. `extra` (paper/deeptex_montage.py's DeeptexSheet)
    may draw a second sheet from the same compare/ folder afterwards."""
    name = os.path.basename(mat_dir.rstrip("/"))
    cmp_dir = layout.newest_set(mat_dir, args.run)
    if not cmp_dir:
        print("  %-28s SKIP (run postprocess.py first)" % name)
        return False
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
        newest = layout.newest(os.path.join(s, "montage.png") for s in layout.set_dirs(d)
                               if os.path.exists(os.path.join(s, "montage.png")))
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
        dirs = layout.materials(args.root, layout.SET_PARENT)
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
