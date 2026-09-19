#!/usr/bin/env python3
"""Metrics shared by both methods. numpy + cv2 only, so dataprep and deeptex agree.

Two halves:

  sharpness() / corr()   Moved here VERBATIM from scripts/compare/matup_postprocess.py:185-196.
                         The committed MatUp results under runs/montages/ were
                         produced by that code, so these must stay bit-identical -
                         see the warning in docs/RUNBOOK.md about refactors that
                         silently move published numbers.

  psnr() / ssim()        New. The comparison chain had NO full-reference metric at
                         all until now: matup_postprocess.py --metrics computes only
                         reference-free sharpness plus one normal-channel
                         correlation. That was the right call while MatUp was the
                         only method, because "sharper" was the whole question.
                         With a second method producing the same baseColor and the
                         same render against the same GT, error against GT becomes
                         measurable and has to be measured.

Deliberately NOT importing DeepTextures' own metric code:

  modules/utils.py:329-333  calculate_ssim's 3-channel branch loops
                            "for _ in range(3): ssims.append(_ssim(img1, img2))" -
                            it averages the same whole-array SSIM three times
                            instead of indexing img1[:,:,i]. Returns garbage on any
                            RGB input.
  modules/utils.py:340-365  rgb2ycbcr mutates its argument in place (img *= 255. at
                            :350) and reverses channels at :351, so despite the
                            name it expects BGR.

Quantisation parity (see the fairness notes in the plan): MatUp writes 16-bit PNGs,
DeepTextures writes 8-bit. Comparing them at 16-bit would hand MatUp precision the
other method structurally cannot have, so psnr/ssim round BOTH sides to `bits`
(default 8, the coarser of the two) before measuring.
"""

import cv2
import numpy as np

U16_MAX = 65535


def to_float(arr):
    """[0,1] float64. Same contract as matup_postprocess.py:86-87."""
    return arr.astype(np.float64) / (U16_MAX if arr.dtype == np.uint16 else 255.0)


# ------------------------------------------------- reference-free (moved verbatim)

def sharpness(arr):
    g = to_float(arr)
    g = g.mean(axis=2) if g.ndim == 3 else g
    gx, gy = np.gradient(g)
    return float(np.hypot(gx, gy).mean()), float(cv2.Laplacian(g, cv2.CV_64F).var())


def corr(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------- full-reference (new)

def _quantised(arr, bits):
    """Both sides onto the same ladder, as a float array in [0, 2**bits - 1]."""
    peak = float(2 ** bits - 1)
    return np.rint(to_float(arr) * peak).clip(0, peak)


def psnr(a, b, bits=8):
    """PSNR over all channels. Identical images give inf."""
    peak = float(2 ** bits - 1)
    x, y = _quantised(a, bits), _quantised(b, bits)
    if x.shape != y.shape:
        raise ValueError("shape mismatch: %s vs %s" % (x.shape, y.shape))
    mse = float(np.mean((x - y) ** 2))
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(peak) - 10.0 * np.log10(mse))


def _ssim_plane(x, y, peak):
    """Wang et al. SSIM on one plane. 11x11 Gaussian, sigma 1.5, valid region."""
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    w = np.outer(k, k.transpose())

    mu1 = cv2.filter2D(x, -1, w)[5:-5, 5:-5]
    mu2 = cv2.filter2D(y, -1, w)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1_sq = cv2.filter2D(x ** 2, -1, w)[5:-5, 5:-5] - mu1_sq
    s2_sq = cv2.filter2D(y ** 2, -1, w)[5:-5, 5:-5] - mu2_sq
    s12 = cv2.filter2D(x * y, -1, w)[5:-5, 5:-5] - mu1_mu2

    m = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / \
        ((mu1_sq + mu2_sq + c1) * (s1_sq + s2_sq + c2))
    return float(m.mean())


def ssim(a, b, bits=8):
    """Mean SSIM, averaged over channels INDIVIDUALLY (the upstream bug fixed)."""
    peak = float(2 ** bits - 1)
    x, y = _quantised(a, bits), _quantised(b, bits)
    if x.shape != y.shape:
        raise ValueError("shape mismatch: %s vs %s" % (x.shape, y.shape))
    if min(x.shape[:2]) < 11:
        raise ValueError("image too small for an 11x11 SSIM window: %s" % (x.shape,))
    if x.ndim == 2:
        return _ssim_plane(x, y, peak)
    return float(np.mean([_ssim_plane(x[:, :, i], y[:, :, i], peak)
                          for i in range(x.shape[2])]))


def full_reference(pred, gt, bits=8):
    """Both full-reference metrics as a dict, for dumping into JSON.

    A non-finite value becomes None rather than Infinity/NaN. json.dump writes
    those as the bare tokens `Infinity` / `NaN`, which are NOT valid JSON and make
    a strict parser reject the whole file - and matup_postprocess.py already
    established the convention of null for an undefined quantity (the
    matup_over_bilinear ratios on a constant metallic map). PSNR is infinite
    exactly when the images are identical, so that case is kept as a flag instead
    of being thrown away.
    """
    p, s = psnr(pred, gt, bits), ssim(pred, gt, bits)
    out = {"psnr": p if np.isfinite(p) else None,
           "ssim": s if np.isfinite(s) else None,
           "bits": bits}
    if not np.isfinite(p):
        out["identical"] = bool(np.array_equal(_quantised(pred, bits),
                                               _quantised(gt, bits)))
    return out
