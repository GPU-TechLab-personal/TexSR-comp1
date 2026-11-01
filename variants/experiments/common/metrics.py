"""
Common evaluation protocol (공통 평가 프로토콜), one implementation for every experiment.

Primary, per claimed map, with the metric that fits the map:
    basecolor   PSNR / SSIM (sRGB, 8-bit ladder, shared texsr_metrics code -
                variants/util/texsr_metrics.py, moved from scripts/common/),
                LPIPS-alex (sRGB), CIEDE2000 deltaE
    normal      mean / median angular error (deg), % within 5 and 10 deg
    roughness   MAE
    metallic    MAE and 0/1 accuracy at 0.5
    height      MAE (raw and mean-aligned) and gradient error
Secondary, mandatory: render-space FLIP and LPIPS, averaged over a Fibonacci
set of point lights (render.eval_lights), rendered with MatUp's model.

Everything takes canonical float materials (protocol.py). `border` crops the
same band from prediction and GT (MatUp's valid region is the inner 496^2).
Seam/tiling error: `band_mask` + `masked_pixel_metrics` report the pixelwise
metrics inside vs outside a band around tile boundaries.
"""
import os

import numpy as np
import torch

from variants.util import texsr_metrics
from variants.experiments.common import render as R
from variants.experiments.common.protocol import MAP_NAMES, crop_border, normalize

_LPIPS = {}


def lpips_model(device):
    """LPIPS-alex, built once per device (downloads AlexNet into TORCH_HOME once)."""
    key = str(device)
    if key not in _LPIPS:
        import lpips
        _LPIPS[key] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    return _LPIPS[key]


def _u16(x):
    return np.round(np.clip(x, 0, 1) * 65535).astype(np.uint16)


def lpips_srgb(pred, gt, device):
    """pred/gt HxWx3 in [0,1] (or NxHxWx3). Mean LPIPS."""
    p = torch.from_numpy(np.ascontiguousarray(pred, np.float32))
    g = torch.from_numpy(np.ascontiguousarray(gt, np.float32))
    if p.ndim == 3:
        p, g = p[None], g[None]
    p = p.permute(0, 3, 1, 2).to(device) * 2 - 1
    g = g.permute(0, 3, 1, 2).to(device) * 2 - 1
    with torch.no_grad():
        return float(lpips_model(device)(p, g).mean())


def delta_e(pred, gt):
    from skimage.color import deltaE_ciede2000, rgb2lab
    return float(deltaE_ciede2000(rgb2lab(np.clip(gt, 0, 1)),
                                  rgb2lab(np.clip(pred, 0, 1))).mean())


def angular_error_deg(pred, gt):
    p, g = normalize(pred.astype(np.float64)), normalize(gt.astype(np.float64))
    cross = np.linalg.norm(np.cross(p, g), axis=-1)
    dot = np.sum(p * g, axis=-1)
    return np.degrees(np.arctan2(cross, dot))       # stable near 0, unlike arccos


def grad(h):
    gy, gx = np.gradient(h.astype(np.float64))
    return gx, gy


# ---------------------------------------------------------------- per map

def basecolor_metrics(pred, gt, device, with_lpips=True):
    out = texsr_metrics.full_reference(_u16(pred), _u16(gt), bits=8)
    out = {"psnr": out["psnr"], "ssim": out["ssim"]}
    out["deltaE00"] = delta_e(pred, gt)
    if with_lpips:
        out["lpips"] = lpips_srgb(pred, gt, device)
    return out


def normal_metrics(pred, gt):
    a = angular_error_deg(pred, gt)
    return {"angle_mean": float(a.mean()), "angle_median": float(np.median(a)),
            "within5": float((a < 5).mean()), "within10": float((a < 10).mean())}


def roughness_metrics(pred, gt):
    return {"mae": float(np.abs(pred - gt).mean())}


def metallic_metrics(pred, gt):
    return {"mae": float(np.abs(pred - gt).mean()),
            "acc01": float(((pred > 0.5) == (gt > 0.5)).mean())}


def height_metrics(pred, gt):
    pgx, pgy = grad(pred)
    ggx, ggy = grad(gt)
    return {"mae": float(np.abs(pred - gt).mean()),
            # height from normal integration has a free constant; report both
            "mae_mean_aligned": float(np.abs((pred - pred.mean()) - (gt - gt.mean())).mean()),
            "grad_err": float(np.hypot(pgx - ggx, pgy - ggy).mean())}


MAP_METRICS = {
    "basecolor": None,              # needs device, handled in evaluate()
    "normal": normal_metrics,
    "roughness": roughness_metrics,
    "metallic": metallic_metrics,
    "height": height_metrics,
}


# ------------------------------------------------------------ render space

def render_metrics(pred, gt, device, n_lights=16, border=0, full_size=None, with_lpips=True):
    """FLIP and LPIPS between renders of pred and GT, averaged over lights."""
    import flip_evaluator as flip
    R.init(device)
    lights = R.eval_lights(n_lights, device)
    rp = R.render_material(pred, lights, device, border, full_size)
    rg = R.render_material(gt, lights, device, border, full_size)
    flips, lp = [], []
    for i in range(rp.shape[-1]):
        a, b = rg[..., i].astype(np.float32), rp[..., i].astype(np.float32)
        flips.append(float(flip.evaluate(a, b, "LDR", applyMagma=False)[1]))
    out = {"flip": float(np.mean(flips)), "flip_per_light": flips, "n_lights": len(flips)}
    if with_lpips:
        p = np.moveaxis(rp, -1, 0)
        g = np.moveaxis(rg, -1, 0)
        out["lpips"] = float(np.mean([lpips_srgb(p[i], g[i], device) for i in range(len(p))]))
    return out


# ------------------------------------------------------------------ driver

def evaluate(pred, gt, device, maps=None, border=0, render=True, n_lights=16,
             with_lpips=True, full_size=None):
    """Full protocol for one material. `maps`: the maps the method CLAIMS
    (primary); every map present in `pred` is still reported (보조 자료).

    Missing maps in `pred` are not invented here - the caller must fill them
    (e.g. with bilinear LR) before asking for render metrics, and say so."""
    claimed = list(maps or [m for m in MAP_NAMES if m in pred])
    p = {k: crop_border(v, border) for k, v in pred.items()}
    g = {k: crop_border(v, border) for k, v in gt.items()}
    res = {"claimed": claimed, "border": border, "maps": {}}
    for name in MAP_NAMES:
        if name not in p:
            continue
        if name == "basecolor":
            res["maps"][name] = basecolor_metrics(p[name], g[name], device, with_lpips)
        else:
            res["maps"][name] = MAP_METRICS[name](p[name], g[name])
    if render:
        missing = [m for m in MAP_NAMES if m not in p]
        if missing:
            raise ValueError("render metrics need all maps; missing %s" % missing)
        full = full_size or (p["basecolor"].shape[0] + 2 * border)
        res["render"] = render_metrics(p, g, device, n_lights, border=0, full_size=full,
                                       with_lpips=with_lpips)
    return res


# ----------------------------------------------------------- seam / tiling

def band_mask(size, tile, band, offset=0):
    """True within `band` px of any tile boundary (tiles of `tile` px starting
    at `offset`). Outer image edges are not boundaries."""
    line = np.zeros(size, bool)
    for b in range(offset % tile, size, tile):
        if 0 < b < size:
            # the boundary lies between pixel b-1 and b: band px on each side
            line[max(b - band, 0):min(b + band, size)] = True
    return line[:, None] | line[None, :]


def masked_pixel_metrics(pred, gt, mask):
    """Pixelwise metrics restricted to mask and to its complement."""
    out = {}
    for region, m in (("band", mask), ("interior", ~mask)):
        r = {}
        if "basecolor" in pred:
            d = (np.round(np.clip(pred["basecolor"], 0, 1) * 255)
                 - np.round(np.clip(gt["basecolor"], 0, 1) * 255))[m]
            mse = float(np.mean(d ** 2))
            r["basecolor_psnr"] = None if mse == 0 else float(20 * np.log10(255) - 10 * np.log10(mse))
            r["basecolor_mae"] = float(np.abs(pred["basecolor"] - gt["basecolor"])[m].mean())
        if "normal" in pred:
            r["normal_angle_mean"] = float(angular_error_deg(pred["normal"], gt["normal"])[m].mean())
        for k in ("roughness", "metallic", "height"):
            if k in pred:
                r[k + "_mae"] = float(np.abs(pred[k] - gt[k])[m].mean())
        r["pixels"] = int(m.sum())
        out[region] = r
    return out
