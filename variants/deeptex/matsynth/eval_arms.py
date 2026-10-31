#!/usr/bin/env python3
"""All five maps plus the render, for the arms that are scored on every map.

Runs in the deeptex container - the old `deeptex_compare.py --maps all
--scalar-collapse mean` (run_deeptex.ps1 eval-arms :603-617, eval-render :884,
gate-smoke :1215):

    python /workspace/variants/deeptex/matsynth/eval_arms.py \\
        --cfg /workspace/variants/deeptex/matsynth/esrgan_matsynth.yaml \\
        --ckpt-dir /workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k \\
        --out /workspace/runs/deeptex/eval/matsynth_all

Every map goes through the shared core (variants/deeptex/infer/deeptex_compare.py:
same restore, same newest MatUp run, same encoding via map_encode.to_rgb8, same
manifest). What this file adds is the 1-channel maps, and with them the one
choice the core never makes: collapsing the network's three output channels back
to one.

WHY A COLLAPSE AT ALL (old deeptex_compare.py:133-147). The network saw three
IDENTICAL channels, but conv_last (modules/models.py:109) has three different
filters, so the three outputs are NOT identical and the choice is visible. test.py
never collapses on the WRITE path - tensor2img keeps all three and cv2.imwrite
emits a 3-channel PNG even for a grayscale input. It collapses only for METRICS,
via rgb2ycbcr(..., only_y=True) (modules/utils.py:340-365), i.e. BT.601 luma, and
feeds that function BGR (cv2.imread) when it reverses channels internally and
expects RGB, so the R and B weights end up swapped. For a replicated grayscale
input that swap is invisible, but it is visible on the non-identical OUTPUT
channels, which is exactly what is collapsed here. The GT and MatUp's column are 1
channel, so 1 channel is written; the only question is which weights.

Also the module deeptex_drift.py reads its JOBS / load_as_rgb01 / save_sr /
newest_run from (it is imported there as `deeptex_compare`, the old name).
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.deeptex.infer import deeptex_compare as core  # noqa: E402

# Re-exported for deeptex_drift.py, which used them as deeptex_compare.<name>.
load_as_rgb01 = core.load_as_rgb01
newest_run = core.newest_run

# name -> (source, kind), the old deeptex_compare.py:95-102 in its order. kind
# decides how the map is squeezed into and back out of a 3-channel uint8 network.
JOBS = [
    ("baseColor", "svbrdf", "baseColor.png", "colour"),
    ("normal",    "svbrdf", "normal.png",    "colour"),
    ("roughness", "svbrdf", "roughness.png", "scalar"),
    ("metallic",  "svbrdf", "metallic.png",  "scalar"),
    ("height",    "svbrdf", "height.png",    "scalar"),
    ("render",    "run",    "render_lr.png", "colour"),
]

COLLAPSE = {
    # symmetric - matches the fact that the three input channels were identical
    "mean":  (1 / 3.0, 1 / 3.0, 1 / 3.0),
    # correct BT.601 luma on RGB
    "luma":  (0.299, 0.587, 0.114),
    # what test.py's metric path actually applies, R/B swapped (see above)
    "paper": (0.114, 0.587, 0.299),
}


def save_sr(path, sr, kind, collapse="mean"):
    """Write the super-resolved map back in the shape its column expects.

    Returns (shape, channel_spread) where channel_spread measures how far apart
    the three output channels actually were - if it is ~0 the collapse choice is
    immaterial, and if it is large it has to be stated.
    """
    if kind != "scalar":
        return core.save_colour(path, sr)
    f = sr.astype(np.float64)
    d = f.max(axis=2) - f.min(axis=2)
    spread = {"mean_255": float(d.mean() * 255.0),
              "p99_255": float(np.percentile(d, 99) * 255.0),
              "max_255": float(d.max() * 255.0)}
    w = COLLAPSE[collapse]
    out = np.rint(np.clip(f[:, :, 0] * w[0] + f[:, :, 1] * w[1] + f[:, :, 2] * w[2],
                          0.0, 1.0) * 255.0).astype(np.uint8)
    if not cv2.imwrite(path, out, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError("cv2.imwrite failed for " + path)
    return list(out.shape), spread


def main():
    ap = core.parser(JOBS, "all", __doc__)
    ap.add_argument("--scalar-collapse", default="mean", choices=sorted(COLLAPSE),
                    help="how a 1-channel map's 3 network outputs are collapsed back "
                         "to 1. 'paper' reproduces test.py's metric-path weights")
    args = ap.parse_args()
    core.run(args, core.select(JOBS, args.maps),
             lambda path, sr, kind: save_sr(path, sr, kind, args.scalar_collapse),
             (args.scalar_collapse, COLLAPSE[args.scalar_collapse]))


if __name__ == "__main__":
    main()
