#!/usr/bin/env python3
"""matup_montage with the paper arm's second sheet, montage_deeptex.png.

Run it in the dataprep container (the old `matup_montage.py --all --collect
/workspace/runs/montages`, run_deeptex.ps1 Stage-Eval :526-527):

    python variants/deeptex/paper/deeptex_montage.py --all --collect /workspace/runs/montages

DeepTextures gets a SECOND sheet, not a fifth row (decision 2026-09-19).
`montage_deeptex.png` is 5x2 - LR / bilinear / MatUp / DeepTex / GT over
baseColor and render - and is written only when deeptex_postprocess.py has
actually filled those columns. As a row it was four placeholder cells wide, and
those placeholders advertised a comparison the released artifact cannot support:
external/DeepTextures has ZERO lines that know a map type, and the 300 files it
ships are colour-dominated, so the checkpoint is a colour model. Splitting keeps
montage.png exactly as it was committed and gives DeepTex a sheet where every cell
is full. See docs/DEEPTEX-DEVIATIONS.md 0.5-0.8.

Everything else is variants/matup/baseline/montage.py, run with the DeeptexSheet
hook below. The hook is the old matup_montage.py:53-61 (constants), :173-185 (the
sheet), :201/:208-212 (the copy in --collect) and :230-237/:255 (the note above
the summary table).
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.matup.baseline import montage as mont  # noqa: E402

# The DeepTextures sheet. Separate grid, separate file - see the module docstring
# for why it is not a fifth row of montage.png.
DEEPTEX_COLUMNS = ["baseColor", "render"]
DEEPTEX_ROWS = [("LR", "lr"), ("bilinear", "bilinear"), ("MatUp", "matup"),
                ("DeepTextures", "deeptex"), ("GT", "gt")]
# "DeepTextures" is 129px at the row-label size and the default gutter is 82px of
# usable width, so this sheet gets a wider one rather than a smaller font than its
# sibling rows. Only this sheet: LEFT stays put, so montage.png is unchanged.
DEEPTEX_LEFT = 160


class DeeptexSheet:
    """The `extra` hook matup_montage.process() / collect() call (see there)."""

    def after_montage(self, mat_dir, cmp_dir, size, stamp):
        # Earned, not reserved: the DeepTextures sheet is drawn only once
        # deeptex_compare.py has actually filled one of its two columns. A MatUp-only
        # run writes exactly what it wrote before this split.
        if any(os.path.exists(os.path.join(cmp_dir, c, "deeptex.png"))
               for c in DEEPTEX_COLUMNS):
            dsheet, _, _ = mont.build(mat_dir, cmp_dir, size,
                                      DEEPTEX_COLUMNS, DEEPTEX_ROWS, "",
                                      left=DEEPTEX_LEFT)
            dout = os.path.join(cmp_dir, "montage_deeptex.png")
            if not cv2.imwrite(dout, dsheet):
                raise SystemExit("cv2.imwrite failed for " + dout)
            print("  %-28s %s  %dx%d" % ("  + montage_deeptex.png", stamp,
                                         dsheet.shape[1], dsheet.shape[0]))

    def collect_one(self, cmp_dir, dest, name):
        dsrc = os.path.join(cmp_dir, "montage_deeptex.png")
        if os.path.exists(dsrc):
            cv2.imwrite(os.path.join(dest, name + "_deeptex.png"),
                        cv2.imread(dsrc, cv2.IMREAD_UNCHANGED))
            return 1
        return 0

    def count_note(self, n_deeptex):
        return (" + %d deeptex sheet(s)" % n_deeptex) if n_deeptex else ""

    def note_lines(self, n_deeptex):
        return ([
            "`<material>_deeptex.png` is a separate 5x2 sheet - LR / bilinear / MatUp /",
            "DeepTex / GT over baseColor and render. DeepTextures is a colour model:",
            "its released code has no notion of a map type, and the 300 training files",
            "that ship with it are colour-dominated, so the other four maps are not",
            "part of that comparison. See docs/DEEPTEX-DEVIATIONS.md 0.5-0.8.",
            "",
        ] if n_deeptex else [])


def main():
    mont.run(mont.parser(__doc__).parse_args(), DeeptexSheet())


if __name__ == "__main__":
    main()
