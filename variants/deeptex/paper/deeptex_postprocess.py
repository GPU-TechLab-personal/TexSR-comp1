#!/usr/bin/env python3
"""matup_postprocess with the paper arm's fifth column: DeepTextures on baseColor and render.

Run it in the dataprep container (the old `matup_postprocess.py --all --metrics
--deeptex-root <eval>`, run_deeptex.ps1 Stage-Eval :522-523):

    python variants/deeptex/paper/deeptex_postprocess.py --all --metrics \\
        --deeptex-root /workspace/runs/deeptex/eval/paper

Everything except the column is variants/compare/matup_postprocess.py, run with the
DeeptexColumn hook below; without --deeptex-root this writes exactly what that
file writes. The column itself is old matup_postprocess.py:71-82 (constants),
:303 and :308-313 (per map), :340-345 (metrics), :365-370 (render) and :400-409
(the console line), unchanged in effect.

Why baseColor and render only (old matup_postprocess.py:71-79): not because the
method cannot do more - the paper applies its 3-channel network to all five
channel types - but because the other four would each need a choice the paper does
not specify. A 1-channel GT needs a 1-channel result, the network emits three
non-identical channels, and the collapse upstream uses for metrics (rgb2ycbcr) is
not what it uses to WRITE, while the paper's real metrics (deltaE, FLIP) have no
code in the repo at all. Those cells stay empty and matup_montage.py draws its
placeholder; the all-maps comparison is variants/deeptex/matsynth/.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.compare import matup_postprocess as post  # noqa: E402

DEEPTEX_COLUMN = "deeptex"
DEEPTEX_MAPS = {"baseColor", "render"}
ORDER_WITH_DEEPTEX = ["lr", "bilinear", "matup", "deeptex", "gt"]


class DeeptexColumn:
    """The `extra` hook matup_postprocess.process() calls (see there)."""

    order = ORDER_WITH_DEEPTEX

    def __init__(self, deeptex_root):
        self.root = deeptex_root

    def add_map(self, cols, name, m, border, size):
        dtx_dir = os.path.join(self.root, name)
        if m in DEEPTEX_MAPS:
            p = os.path.join(dtx_dir, m + ".png")
            if os.path.exists(p):
                cols[DEEPTEX_COLUMN] = post.deeptex_column(p, border, size)
            else:
                print("  ! %s: no DeepTextures output for %s (%s)" % (name, m, p))

    def add_render(self, rcols, name, border, side):
        dtx_dir = os.path.join(self.root, name)
        p = os.path.join(dtx_dir, "render.png")
        if os.path.exists(p):
            rcols[DEEPTEX_COLUMN] = post.deeptex_column(p, border, side)
        else:
            print("  ! %s: no DeepTextures render (%s)" % (name, p))

    def add_metrics(self, entry, cols, gb, lb):
        if DEEPTEX_COLUMN in cols:
            gd, ld = post.sharpness(cols[DEEPTEX_COLUMN])
            entry[DEEPTEX_COLUMN] = {"grad": gd, "lapvar": ld}
            entry["deeptex_over_bilinear"] = {
                "grad": gd / gb if gb else None,
                "lapvar": ld / lb if lb else None}

    def ratio_note(self, map_entry, fmt):
        d = map_entry.get("deeptex_over_bilinear")
        if d:
            return "   deeptex/bilinear |grad| %s" % fmt(d["grad"])
        return ""

    def psnr_note(self, v, pf):
        if "deeptex" in v:
            return " / deeptex %s" % pf(v["deeptex"])
        return ""


def main():
    ap = post.parser(__doc__)
    ap.add_argument("--deeptex-root", default="",
                    help="runs/deeptex/eval/<tag>; adds a fifth column for the "
                         "two maps an RGB method can produce (baseColor, render)")
    args = ap.parse_args()
    post.run(args, DeeptexColumn(args.deeptex_root) if args.deeptex_root else None)


if __name__ == "__main__":
    main()
