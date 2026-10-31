#!/usr/bin/env python3
"""
Gate: the LR protocol, augmentation and metrics do what their docstrings say.
Seconds, GPU optional.

    python variants/experiments/gates/gate_protocol.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.experiments.common import protocol as P   # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print("  %-58s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILS.append(name)


def normals_from_height(h, strength=4.0):
    """OpenGL: +Y up = -row. n ~ (-dh/dx, -dh/dy_up, 1) = (-h_col, +h_row, 1)."""
    h_row, h_col = np.gradient(h * strength)
    return P.normalize(np.dstack([-h_col, h_row, np.ones_like(h)]))


def material(size=64, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.random((max(size // 8, 2), max(size // 8, 2)))
    import cv2
    h = cv2.resize(h, (size, size), interpolation=cv2.INTER_CUBIC)
    return {"basecolor": rng.random((size, size, 3)).astype(np.float32),
            "normal": normals_from_height(h).astype(np.float32),
            "roughness": rng.random((size, size)).astype(np.float32),
            "metallic": (rng.random((size, size)) > 0.5).astype(np.float32),
            "height": ((h - h.min()) / np.ptp(h)).astype(np.float32)}


def main():
    print("protocol")
    # basecolor is averaged in linear light: a black/white checker is NOT 0.5 sRGB
    chk = np.indices((4, 4)).sum(0) % 2
    m = material(4)
    m["basecolor"] = np.dstack([chk] * 3).astype(np.float32)
    lr = P.downsample(m, 4)
    expect = float(P.linear_to_srgb(0.5))
    check("basecolor block mean is linear-light", abs(lr["basecolor"][0, 0, 0] - expect) < 1e-6,
          "%.4f vs %.4f" % (lr["basecolor"][0, 0, 0], expect))

    m = material(64)
    lr = P.downsample(m, 4)
    check("normals renormalised", np.allclose(np.linalg.norm(lr["normal"], axis=-1), 1, atol=1e-5))

    flat = material(16)
    flat["normal"][:] = [0, 0, 1]
    flat["roughness"][:] = 0.3
    for mode in P.ROUGHNESS_MODES:
        r = P.downsample(flat, 4, mode)["roughness"]
        check("flat normals keep constant roughness (%s)" % mode, np.allclose(r, 0.3, atol=1e-6))

    bumpy = material(64, 1)
    bumpy["roughness"][:] = 0.2
    rl = P.downsample(bumpy, 4, "lean")["roughness"]
    rt = P.downsample(bumpy, 4, "toksvig")["roughness"]
    check("lean/toksvig raise roughness under normal spread", (rl >= 0.2 - 1e-6).all() and (rt >= 0.2 - 1e-6).all(),
          "mean lean %.3f toksvig %.3f" % (rl.mean(), rt.mean()))
    check("lean ~ toksvig to first order", abs(rl.mean() - rt.mean()) < 0.05,
          "diff %.4f" % abs(rl.mean() - rt.mean()))

    # composition: basecolor/metallic/height area means compose exactly
    a = P.downsample(P.downsample(m, 2), 2)
    b = P.downsample(m, 4)
    check("area maps compose (2x2 == 4)", all(np.allclose(a[k], b[k], atol=1e-5)
                                             for k in ("basecolor", "metallic", "height")))

    print("augmentation")
    import cv2
    rng = np.random.default_rng(3)
    h = cv2.resize(rng.random((8, 8)), (64, 64), interpolation=cv2.INTER_CUBIC)
    base = {"basecolor": np.zeros((64, 64, 3), np.float32), "roughness": np.zeros((64, 64), np.float32),
            "metallic": np.zeros((64, 64), np.float32), "height": h.astype(np.float32),
            "normal": normals_from_height(h).astype(np.float32)}
    for k in range(4):
        for flip in (False, True):
            got = P.flip_rot(base, k, flip)
            hh = np.rot90(h, k)
            hh = hh[:, ::-1] if flip else hh
            want = normals_from_height(np.ascontiguousarray(hh))
            err = np.degrees(np.arccos(np.clip((got["normal"] * want).sum(-1), -1, 1)))[2:-2, 2:-2]
            check("flip_rot normal k=%d flip=%s" % (k, flip), err.max() < 0.5, "max %.3f deg" % err.max())

    print("io")
    with tempfile.TemporaryDirectory() as d:
        P.save_material(m, d)
        back = P.load_material(d)
        worst = max(float(np.abs(back[k] - (P.normalize(m[k]) if k == "normal" else m[k])).max()) for k in m)
        check("save/load round trip (16-bit)", worst < 1e-4, "max %.2e" % worst)

    print("metrics")
    from variants.experiments.common import metrics as M
    mask = M.band_mask(64, 16, 2)
    check("band mask: 3 interior boundaries x 4px x 2 axes",
          mask[0].sum() == 12 and mask[16].sum() == 64 and not mask[5, 5] and not mask[13, 5]
          and mask[14, 5] and mask[17, 5] and not mask[18, 5],
          "%d px" % mask.sum())
    import torch
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    big = material(64, 5)
    r = M.evaluate(big, big, dev, with_lpips=True, n_lights=4)
    mp = r["maps"]
    check("identical -> psnr None (inf)", mp["basecolor"]["psnr"] is None)
    check("identical -> zero errors",
          mp["normal"]["angle_mean"] < 1e-3 and mp["roughness"]["mae"] == 0 and mp["height"]["mae"] == 0
          and mp["metallic"]["acc01"] == 1.0 and mp["basecolor"]["deltaE00"] < 1e-6,
          "angle %.2e" % mp["normal"]["angle_mean"])
    check("identical -> FLIP 0, LPIPS 0", r["render"]["flip"] < 1e-6 and r["render"]["lpips"] < 1e-6,
          "flip %.2e lpips %.2e" % (r["render"]["flip"], r["render"]["lpips"]))
    worse = dict(big, normal=P.normalize(big["normal"] + np.array([0.3, 0, 0], np.float32)))
    r2 = M.evaluate(worse, big, dev, n_lights=4)
    check("perturbed normal -> render error > 0", r2["render"]["flip"] > 0.01,
          "flip %.3f angle %.1f" % (r2["render"]["flip"], r2["maps"]["normal"]["angle_mean"]))

    print("\n%s" % ("ALL PASS" if not FAILS else "FAILED: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
