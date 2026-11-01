"""Image I/O and the conversions every comparison column goes through.

Only cv2 and numpy: PIL silently truncates 16-bit RGB PNGs to 8-bit. Columns are
kept as 16-bit (to_u16) so 8-bit and 16-bit producers share a dtype; display
conversion is >> 8 with no per-cell rescaling, so an all-zero map stays black.
"""
import os

import cv2
import numpy as np

U16_MAX = 65535


def read(path):
    """cv2.imread(IMREAD_UNCHANGED); a missing or unreadable file is fatal."""
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise SystemExit("cannot read " + path)
    return arr


def read_if(path):
    """cv2.imread(IMREAD_UNCHANGED); None when the file is absent or unreadable."""
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def write(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, arr):
        # cv2.imwrite returns False instead of raising - a missing directory or a
        # bad path fails silently otherwise.
        raise SystemExit("cv2.imwrite failed for " + path)


def to_u16(arr):
    """Promote to 16-bit so every column shares a dtype. 257 = 65535/255."""
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        return arr.astype(np.uint16) * 257
    raise SystemExit("unexpected dtype %s" % arr.dtype)


def to_float(arr):
    return arr.astype(np.float64) / (U16_MAX if arr.dtype == np.uint16 else 255.0)


def to_display(arr, size, drop_alpha=False):
    """16-bit or 8-bit, gray or colour -> 8-bit BGR at `size`, no rescaling.
    `drop_alpha` keeps the first three channels of a 4-channel image."""
    if arr is None:
        return None
    a = (arr >> 8).astype(np.uint8) if arr.dtype == np.uint16 else arr.astype(np.uint8)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    interp = cv2.INTER_AREA if a.shape[0] > size else cv2.INTER_NEAREST
    if drop_alpha:
        a = a[:, :, :3]
    return cv2.resize(a, (size, size), interpolation=interp)


def fit_to(a, border, side):
    """Bring one producer's image onto the common valid region: crop `border`
    pixels off each side when it is larger than `side`, resize (nearest) to
    `side` when it still differs, promote to 16-bit.

    Use it for a method whose whole output is valid next to one that has an
    unoptimised rim (MatUp's 8px): cropping both is the only way the columns
    cover the same pixels of the same material.
    """
    if a.shape[0] > side and border:
        a = a[border:-border, border:-border]
    if a.shape[0] != side:
        a = cv2.resize(a, (side, side), interpolation=cv2.INTER_NEAREST)
    return to_u16(a)


def load_fit(path, border, side):
    """read() + fit_to(): one output PNG as a comparison column."""
    return fit_to(read(path), border, side)
