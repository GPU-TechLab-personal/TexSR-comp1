#!/usr/bin/env python3
"""Score the tiling run and measure the seams the paper says are not there.

Runs in the dataprep container (numpy + cv2 only; no TensorFlow).

WHY THIS EXISTS
---------------
`external/DeepTextures/test.py` is the only part of the released artifact that
implements the paper's "scalable tiling", and it is the only part of the paper's
own pipeline our 128 -> 512 comparison never exercises (split_large_image fires
only when a side exceeds 512; docs/DEEPTEX-DEVIATIONS.md 0.8). This script runs
the scoring for a separate experiment that DOES exercise it, on held-out images
from the paper's own textures_db.

It compares three reconstructions of the same ground truth:

  bicubic   imresize_np x4 of the LR                    (the paper's own baseline)
  upstream  test.py verbatim: 512 tiles -> JPEG q95 -> per-tile LR -> SR -> butt-join
  ours      deeptex_sr.py: LR-space tiles with overlap, in memory, no JPEG

and it scores them with ONE metric implementation (scripts/common/texsr_metrics),
which matters: test.py prints PSNR/SSIM computed through rgb2ycbcr fed a BGR
array, so its red and blue luma weights are swapped (modules/utils.py:351,355).
Those numbers are internally consistent but are not comparable to anything else,
including each other across runs. Everything here is recomputed from the PNGs.

THE SEAM MEASUREMENT
--------------------
The paper says tiles are "processed and reassembled seamlessly"; Fig. 6 says the
patches are "delimited" with "no overlap", and grep finds no seam, blend, overlap
or feather handling anywhere in the submodule. join_large_image pastes each tile
into a fixed 512 slot (modules/utils.py:428-435), so any disagreement between two
neighbouring tiles lands on one pixel column.

So: take the mean absolute first difference ACROSS each interior tile boundary,
and divide by the same quantity averaged over every non-boundary position. A
seamless join gives ~1.0 - the boundary is statistically ordinary. Anything
meaningfully above 1.0 is a visible edge, and the ratio says how much stronger it
is than the image's own local contrast.

The ground truth is measured the same way, and that calibration turns out to be
essential: some textures have real structure on a 512 grid, so a raw ratio above
1.0 can be content rather than a seam. The number to read is therefore
`seam_vs_gt` = ratio(image) / ratio(ground truth), which divides the content out.
1.0 means the join is as smooth as the original at the same pixels.

CAVEAT ON THE `ours` COLUMN. This measurement is tied to upstream's fixed 512
grid. deeptex_sr.py tiles in LR space with an overlap, so its boundaries fall
somewhere else entirely and mostly miss the columns sampled here. The `ours`
seam figure is therefore NOT a like-for-like reading and is reported only to show
that nothing pathological happens at those columns; do not compare it directly
with the upstream figure.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.common import paths  # noqa: E402
from variants.common import texsr_metrics  # noqa: E402

TILE = 512  # split_large_image's hardcoded grid (modules/utils.py:376-395)

# What goes ON the image. `upstream` and `ours` are internal names for which
# tiling implementation ran - they say nothing to a reader, and "ours" hides that
# BOTH arms are the paper's method with the paper's checkpoint. Only the join
# differs, so the label says that. ASCII only: cv2's Hershey fonts have no Hangul.
DISPLAY = {
    "bicubic":  "Bicubic",
    "upstream": "DeepTextures - paper tiling",
    "ours":     "DeepTextures - overlap tiling",
    "GT":       "Ground truth",
}


def read(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise SystemExit("cannot read " + path)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    return a[:, :, :3]


def seam_ratio(img, tile=TILE):
    """How much stronger the tile boundaries are than ordinary adjacent columns.

    Returns None when the image has no interior boundary to measure.
    """
    a = img.astype(np.float64)
    h, w = a.shape[:2]
    dx = np.abs(np.diff(a, axis=1)).mean(axis=(0, 2))   # per column gap
    dy = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2))   # per row gap

    # np.diff index i is the gap between i and i+1, so the gap that straddles a
    # boundary at x = k*tile sits at index k*tile - 1.
    bx = [k * tile - 1 for k in range(1, w // tile) if 0 <= k * tile - 1 < len(dx)]
    by = [k * tile - 1 for k in range(1, h // tile) if 0 <= k * tile - 1 < len(dy)]
    if not bx and not by:
        return None

    seam = np.concatenate([dx[bx], dy[by]]) if (bx and by) else \
        (dx[bx] if bx else dy[by])
    mask_x = np.ones(len(dx), bool); mask_x[bx] = False
    mask_y = np.ones(len(dy), bool); mask_y[by] = False
    ordinary = np.concatenate([dx[mask_x], dy[mask_y]])

    base = float(ordinary.mean())
    return {"seam_mean": float(seam.mean()),
            "ordinary_mean": base,
            "ratio": (float(seam.mean()) / base) if base > 0 else None,
            "boundaries": len(seam)}


def label(img, text, size):
    cell = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(cell, (0, 0), (size, 22), (18, 18, 18), -1)
    scale = 0.45
    while cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > size - 12             and scale > 0.28:
        scale -= 0.02
    cv2.putText(cell, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (235, 235, 235), 1, cv2.LINE_AA)
    return cell


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/workspace/runs/deeptex/tiling/src")
    ap.add_argument("--upstream", default="/workspace/runs/deeptex/tiling/upstream")
    ap.add_argument("--ours", default="/workspace/runs/deeptex/tiling/ours")
    ap.add_argument("--bicubic", default="/workspace/runs/deeptex/tiling/bicubic")
    ap.add_argument("--out", default="/workspace/runs/deeptex/tiling")
    ap.add_argument("--cell", type=int, default=340)
    args = ap.parse_args()

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.src)
                   if f.lower().endswith(".png"))
    report = {"tile": TILE, "materials": {}}

    for stem in stems:
        gt = read(os.path.join(args.src, stem + ".png"))
        # test.py names the joined mosaic HR_<basename>.png, where <basename>
        # already carries the .png extension (test.py:76).
        up_path = os.path.join(args.upstream, stem, "HR_%s.png.png" % stem)
        if not os.path.exists(up_path):                       # 1x1 grid: no join
            up_path = os.path.join(args.upstream, stem, "HR_%s.png_0_0.png" % stem)
        cols = {"bicubic": os.path.join(args.bicubic, stem + ".png"),
                "upstream": up_path,
                "ours": os.path.join(args.ours, stem + ".png")}

        entry = {"gt_shape": list(gt.shape),
                 "grid": "%dx%d" % (gt.shape[0] // TILE, gt.shape[1] // TILE),
                 "seam": {"gt": seam_ratio(gt)}, "quality": {}}
        imgs = {"GT": gt}

        for name, path in cols.items():
            if not os.path.exists(path):
                entry["quality"][name] = None
                continue
            a = read(path)
            if a.shape[:2] != gt.shape[:2]:
                entry["quality"][name] = {"error": "shape %s vs GT %s"
                                          % (a.shape[:2], gt.shape[:2])}
                continue
            imgs[name] = a
            entry["quality"][name] = texsr_metrics.full_reference(a, gt)
            entry["seam"][name] = seam_ratio(a)

        # Divide the content out: a texture whose pattern lines up with the 512
        # grid raises every method's raw ratio, ground truth included.
        gt_ratio = (entry["seam"].get("gt") or {}).get("ratio")
        for k in ("upstream", "ours"):
            v = entry["seam"].get(k)
            if v and v.get("ratio") is not None and gt_ratio:
                v["seam_vs_gt"] = v["ratio"] / gt_ratio
        report["materials"][stem] = entry

        order = ["bicubic", "upstream", "ours", "GT"]
        cells = [label(imgs[k], DISPLAY[k], args.cell) for k in order if k in imgs]
        cv2.imwrite(os.path.join(args.out, "sheet_%s.png" % stem), np.hstack(cells))

        # A close-up straddling the first interior boundary, upstream vs GT.
        b = TILE
        if gt.shape[1] > b and "upstream" in imgs:
            half, y0 = 96, max(0, gt.shape[0] // 2 - 96)
            crop = lambda im: im[y0:y0 + 192, b - half:b + half]
            zoom = lambda im: cv2.resize(crop(im), (args.cell, args.cell),
                                         interpolation=cv2.INTER_NEAREST)
            pair = [label(zoom(imgs[k]), "%s   x=%d" % (DISPLAY[k], b), args.cell)
                    for k in ("upstream", "ours", "GT") if k in imgs]
            cv2.imwrite(os.path.join(args.out, "seam_%s.png" % stem), np.hstack(pair))

        q = entry["quality"]
        def fmt(k):
            v = q.get(k)
            return "  n/a " if not v or "psnr" not in v else "%6.2f" % v["psnr"]
        u = entry["seam"].get("upstream") or {}
        print("  %-6s %-5s  PSNR bic %s upstream %s ours %s   seam vs GT %s"
              % (stem, entry["grid"], fmt("bicubic"), fmt("upstream"), fmt("ours"),
                 "n/a" if u.get("seam_vs_gt") is None else "%5.2fx" % u["seam_vs_gt"]))

    with open(os.path.join(args.out, "tiling_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("\n%d image(s) -> %s/tiling_report.json" % (len(stems), args.out))


if __name__ == "__main__":
    main()
