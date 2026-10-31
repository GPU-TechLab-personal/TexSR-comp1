"""Which MatUp run a material's columns come from: the one copy of the rule.

The old tree had it four times - deeptex_compare.py:183-194,
svbrdf_render_eval.py:450-456, deeptex_arms_report.py:104-109 as functions, and
inline in matup_postprocess.py:260-267 - plus the copies this tree does not own
(render_gate.py:378-386, and the inline ones in matup_render.py:165-172 and
matup_height.py:113-115, which are moved verbatim). All of them are
`sorted(...)[-1]` over the isdir() entries of <mat>/matup-output-<mat>/, filtered to
one stamp when asked; following it keeps every DeepTextures column aligned with
the MatUp run the other columns came from.
"""

import glob
import os


def newest_run(mat_dir, name, want):
    """sorted(...)[-1], the chain-wide rule."""
    runs = sorted(d for d in glob.glob(os.path.join(mat_dir, "matup-output-%s" % name, "*"))
                  if os.path.isdir(d))
    if want:
        runs = [r for r in runs if os.path.basename(r) == want]
    return runs[-1] if runs else None
