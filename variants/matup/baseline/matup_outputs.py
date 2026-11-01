"""How to read a MatUp run and turn it into comparison columns.

A material folder (variants/util/matsynth_extract.py + run.ps1) is

    <root>/<m>/SVBRDF/<map>.png                    the 128^2 LR input
    <root>/<m>/GT/<map>.png                        the 512^2 ground truth
    <root>/<m>/matup-output-<m>/<stamp>/<map>_opt.png, height_opt.png, render_*.png

Three things have to be normalised before MatUp output can stand next to the
ground truth, or differences that have nothing to do with quality get read as
quality differences:

* **The 8px black border.** The paper optimises only over valid patches ("we use
  a sliding window mechanism over valid patches to avoid the borders"), so the
  rim is absent prediction padded with zeros, not data. Its width is
  `tile_width // 2`, not a constant 8 - change --tile_width and it moves.
  detect_border() measures it from the images.
* **The normal map's green channel.** `matup/data_utils.py` calls `write_normal`
  with a hardcoded `opengl_normals=False`, so the output is always in the
  internal (DirectX, +Y down) convention. MatSynth is OpenGL, so the two
  disagree and bumps read as dents (measured Y correlation against GT: -0.986 on
  the brick, -0.809 on the foam; flipping G makes both positive).
* **Size.** LR is 128^2, everything else 512^2.

build_columns() returns the four columns every comparison is built on: lr
(nearest, a view of the actual input), bilinear (MatUp's own initialisation,
cv2.INTER_LINEAR x4 of the whole LR map, then cropped), matup, gt.
"""
import glob
import os

import cv2
import numpy as np

from variants.util.report.images import U16_MAX, to_u16

# map name -> (LR/GT filename, MatUp output filename)
MAPS = {
    "baseColor": ("baseColor.png", "baseColor_opt.png"),
    "normal":    ("normal.png",    "normal_opt.png"),
    "roughness": ("roughness.png", "roughness_opt.png"),
    "metallic":  ("metallic.png",  "metallic_opt.png"),
    # matup_height.py reconstructs this (paper 3.3); MatUp itself writes none.
    # It comes out already cropped to the valid region.
    "height":    ("height.png",    "height_opt.png"),
}
PRECROPPED = {"height"}

# matup_render.py renders all of these with MatUp's own renderer and eval light.
# The bilinear one is MatUp's own initialisation rendered, which is what makes it
# the baseline the paper compares against (Fig. 8).
RENDER = {"lr": "render_lr.png", "bilinear": "render_bilinear.png",
          "matup": "render_opt.png", "gt": "render_gt.png"}
PRECROPPED_RENDER = {"gt", "bilinear"}   # both are produced at 496^2 already

COLUMNS = ["lr", "bilinear", "matup", "gt"]


def newest_run(mat_dir, name, want):
    """The MatUp run a material's columns come from: sorted(...)[-1] over
    <mat>/matup-output-<mat>/*, or the one stamped `want`. Every report follows
    it, which keeps a DeepTextures column aligned with the MatUp run beside it."""
    runs = sorted(d for d in glob.glob(os.path.join(mat_dir, "matup-output-%s" % name, "*"))
                  if os.path.isdir(d))
    if want:
        runs = [r for r in runs if os.path.basename(r) == want]
    return runs[-1] if runs else None


def detect_border(opt_arrays):
    """Count the zero rim MatUp padded on, from the images themselves.

    A row counts as border only when it is zero in *every* map passed in. Some
    materials have an all-zero metallic (the brick does), so testing one map
    alone would swallow the whole image.

    Callers must NOT pass the normal map. It is written through `write_normal`,
    which encodes as `65535 * (0.5 * n + 0.5)`, so its zero-padded rim comes out
    at 32767 rather than 0 and would hide the border entirely.
    """
    peak = None
    for arr in opt_arrays:
        m = arr.max(axis=2) if arr.ndim == 3 else arr
        peak = m if peak is None else np.maximum(peak, m)

    h, w = peak.shape
    top = bottom = left = right = 0
    while top < h and peak[top].max() == 0:
        top += 1
    while bottom < h and peak[h - 1 - bottom].max() == 0:
        bottom += 1
    while left < w and peak[:, left].max() == 0:
        left += 1
    while right < w and peak[:, w - 1 - right].max() == 0:
        right += 1
    return min(top, bottom, left, right), (top, bottom, left, right)


def build_columns(lr, gt, opt, border, scale, is_normal, opt_precropped=False):
    """Crop all four views to the same valid region at the same size."""
    hr = gt.shape[0]
    b = border
    lb = b // scale

    # a[b:-b] degenerates to an empty slice when b == 0, so guard it rather than
    # writing empty PNGs (cv2.imwrite asserts instead of returning False).
    crop_hr = (lambda a: a[b:-b, b:-b]) if b else (lambda a: a)

    # bilinear must upsample the WHOLE LR map and only then crop, so that it is
    # bit-for-bit the operation MatUp initialises from.
    bilinear = crop_hr(cv2.resize(lr, (hr, hr), interpolation=cv2.INTER_LINEAR))

    # nearest for the LR column on purpose: bilinear here would make it a second
    # bilinear result rather than a view of the actual input.
    lr_view = lr[lb:-lb, lb:-lb] if lb else lr
    side = (hr - 2 * b)
    lr_view = cv2.resize(lr_view, (side, side), interpolation=cv2.INTER_NEAREST)

    matup = (opt if opt_precropped else crop_hr(opt)).copy()
    if is_normal:
        m = to_u16(matup)
        m[..., 1] = U16_MAX - m[..., 1]      # DirectX -> OpenGL, value inversion
        matup = m

    return {
        "lr":       to_u16(lr_view),
        "bilinear": to_u16(bilinear),
        "matup":    to_u16(matup),
        "gt":       to_u16(crop_hr(gt)),
    }
