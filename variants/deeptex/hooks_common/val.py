"""Held-out validation of the old trainer (deeptex_train.py:377-433), moved unchanged.

Neither repo has a dataset-level evaluation; --val-every adds one against the
held-out split. Used by the fixed arm; the render arm runs with --no-val (as
run_deeptex.ps1 always ran it) but keeps the same code path for parity.
"""

import math
import os

import cv2
import numpy as np
import tensorflow as tf

from variants.common import texsr_metrics


def load_val_pairs(cfg, limit):
    """Held-out tiles as (name, lr_rgb_uint8, hr_rgb_uint8).

    cv2 gives BGR; training fed RGB via tf.image.decode_png (modules/dataset.py:12-13).
    Converting here is the same bug test.py:57,68,70 has and never fixed - it feeds
    cv2's BGR straight into a model trained on RGB.
    """
    vc = cfg.get("val_dataset") or {}
    hr_dir, lr_dir = vc.get("hr_dir"), vc.get("lr_dir")
    if not hr_dir or not os.path.isdir(hr_dir) or not os.path.isdir(lr_dir or ""):
        return []
    names = sorted(f for f in os.listdir(hr_dir) if f.lower().endswith(".png"))
    if limit:
        # Even stride, so the sample spans sources rather than taking one source's
        # first N tiles.
        step = max(1, len(names) // limit)
        names = names[::step][:limit]
    out = []
    for n in names:
        lr = cv2.imread(os.path.join(lr_dir, n), cv2.IMREAD_UNCHANGED)
        hr = cv2.imread(os.path.join(hr_dir, n), cv2.IMREAD_UNCHANGED)
        if lr is None or hr is None:
            continue
        out.append((n, cv2.cvtColor(lr, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)))
    return out


def validate(generator, pairs, scale, with_baseline):
    """Mean PSNR/SSIM over the held-out tiles, plus a bicubic baseline to beat.

    `generator` must be the DYNAMIC-shape model. The training model is built as
    RRDB_Model(cfg['input_size'], ...) = Input([32, 32, 3]), so Keras rejects a
    120x120 validation tile outright even though the architecture is fully
    convolutional. models.py:98-99 switches to tf.shape(fea) only when size is
    None, which is exactly why test.py:33 builds RRDB_Model(None, ...).
    """
    ps, ss, bps = [], [], []
    for _, lr, hr in pairs:
        x = tf.convert_to_tensor(lr[None].astype(np.float32) / 255.0)
        sr = generator(x, training=False).numpy()[0]
        sr = np.rint(np.clip(sr, 0, 1) * 255).astype(np.uint8)
        h, w = hr.shape[:2]
        sr = sr[:h, :w]
        ps.append(texsr_metrics.psnr(sr, hr))
        ss.append(texsr_metrics.ssim(sr, hr))
        if with_baseline:
            bic = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
            bps.append(texsr_metrics.psnr(bic, hr))
    finite = [p for p in ps if math.isfinite(p)]
    res = {"n": len(ps),
           "psnr": float(np.mean(finite)) if finite else float("nan"),
           "ssim": float(np.mean(ss)) if ss else float("nan")}
    if bps:
        fb = [p for p in bps if math.isfinite(p)]
        res["bicubic_psnr"] = float(np.mean(fb)) if fb else float("nan")
    return res
