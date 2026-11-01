"""What a comparison set measures: reference-free sharpness, a correlation, and
PSNR/SSIM of every column against GT.

The full-reference numbers come from variants/util/texsr_metrics.py, so every
method is scored by the same code. sharpness() / corr() produced the committed
runs/montages/ numbers and must stay as they are.
"""
import cv2
import numpy as np

from variants.util import texsr_metrics
from variants.util.report.images import to_float


def sharpness(arr):
    """(mean gradient magnitude, Laplacian variance) of the luminance-free mean."""
    g = to_float(arr)
    g = g.mean(axis=2) if g.ndim == 3 else g
    gx, gy = np.gradient(g)
    return float(np.hypot(gx, gy).mean()), float(cv2.Laplacian(g, cv2.CV_64F).var())


def corr(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ratio(a, b):
    """a / b, or None when b is 0 (a constant map has no gradient to divide by)."""
    return a / b if b else None


def full_reference(cols, gt_key="gt"):
    """PSNR/SSIM of every non-GT column against GT.

    Both sides are quantised to 8 bits first (texsr_metrics.full_reference): one
    producer may write 16-bit and another 8-bit, and scoring at 16 would hand the
    first a precision the second structurally cannot have.
    """
    out = {}
    for col, arr in cols.items():
        if col == gt_key:
            continue
        try:
            out[col] = texsr_metrics.full_reference(arr, cols[gt_key])
        except ValueError as e:
            out[col] = {"error": str(e)}
    return out
