#!/usr/bin/env python3
"""
Reconstruct the height map MatUp's implementation never writes, per paper 3.3.

    docker compose run --rm dataprep python variants/matup/baseline/matup_height.py --all

The paper:

    "most of the geometric mesostructures are captured by the normal map ...
     while reconstructing the height map from it afterwards ... we adopt the
     normal integration scheme proposed in [DC07], which requires a Poisson
     solve. As noticed by Nehab et al., geometry reconstruction by normal
     integration suffers from low-frequency noise. Fortunately, the
     low-resolution input height map provides us with a dense enough anchor to
     combat this, by compositing a high-pass filtering of the upsampled
     normal-based height channel with a bilinear upsampling of the input."

The reference implementation has none of it - `matup/data_utils.py` writes five
files and height is not among them, and a repo-wide search for poisson/integrat/
frankot turns up nothing. So this runs as a post-process, outside the pinned
submodule.

Three details that decide whether the result is meaningful:

* **Only the valid region is integrated.** MatUp zero-pads its output, and
  `write_normal` encodes as `65535*(0.5n+0.5)`, so the border decodes to
  (0,0,0) - not a unit normal. Integrating across it diverges at the edges.

* **Neumann, not periodic.** The 496^2 valid crop of a tileable 512^2 material is
  no longer tileable, so a DCT (reflecting) solve is right and an FFT (wrapping)
  one would fight a seam that is not really there.

* **The scale is derived, not tuned.** The XY grid spans [-1,1] across the full
  output width, so one pixel is 2/size in world units; and MatUp displaces by
  `disp_val * (h - 0.5)` with disp_val = 0.01 hardcoded at dataLoader.py:154.
  Both conversions are therefore fixed, not fitted.
"""
import argparse
import glob
import os

import cv2
import numpy as np
from scipy.fft import dctn, idctn

DISP_VAL = 0.01          # dataLoader.py:154
U16_MAX = 65535


def read(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise SystemExit("cannot read " + path)
    return a


def to_float(a):
    return a.astype(np.float64) / (U16_MAX if a.dtype == np.uint16 else 255.0)


def unpack_normal_opengl(arr):
    """MatUp output -> unit normals in the OpenGL convention (+Y up).

    `matup/data_utils.py` calls write_normal with a hardcoded
    opengl_normals=False, so the file is always in the internal DirectX
    convention regardless of how the input was read. Flip G back.
    """
    n = to_float(arr)[..., ::-1]        # cv2 gives BGR; we want RGB = XYZ
    n = 2.0 * n - 1.0
    n[..., 1] *= -1.0                   # DirectX -> OpenGL
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.clip(norm, 1e-8, None)


def poisson_dct(div):
    """Solve laplacian(h) = div with Neumann boundaries, via DCT-II."""
    h, w = div.shape
    f = dctn(div, type=2, norm="ortho")
    yy = np.arange(h).reshape(-1, 1)
    xx = np.arange(w).reshape(1, -1)
    denom = 2.0 * (np.cos(np.pi * yy / h) - 1.0) + 2.0 * (np.cos(np.pi * xx / w) - 1.0)
    denom[0, 0] = 1.0                   # the constant term is unconstrained
    f = f / denom
    f[0, 0] = 0.0                       # pin the mean; the anchor supplies it
    return idctn(f, type=2, norm="ortho")


def integrate(normal):
    """Unit normals -> height, in the same world units as the XY grid.

    The y sign is NOT symmetric with x. OpenGL normals point +Y up while the
    row index runs down, so the row derivative picks up the opposite sign.
    Measured on the MatSynth ground truth pair (GT/height.png vs GT/normal.png)
    for 0559_brick_uneven_stones:

        nx vs -dh/dcol  +0.978        ny vs -dh/drow  -0.976
                                      ny vs +dh/drow  +0.976

    so dh/dcol = -nx/nz but dh/drow = +ny/nz.
    """
    nx, ny, nz = normal[..., 0], normal[..., 1], normal[..., 2]
    nz = np.clip(nz, 1e-3, None)
    p, q = -nx / nz, ny / nz

    # div(p, q) with the same one-sided differences at the edges numpy uses
    dpdx = np.gradient(p, axis=1)
    dqdy = np.gradient(q, axis=0)
    return poisson_dct(dpdx + dqdy)


def process(mat_dir, args):
    name = os.path.basename(mat_dir.rstrip("/"))
    runs = sorted(d for d in glob.glob(
        os.path.join(mat_dir, "matup-output-%s" % name, "*")) if os.path.isdir(d))
    if args.run:
        runs = [d for d in runs if os.path.basename(d) == args.run]
    if not runs:
        print("  %-28s SKIP (no MatUp output)" % name)
        return False
    run_dir = runs[-1]

    b = args.tile_width // 2
    normal_full = read(os.path.join(run_dir, "normal_opt.png"))
    size = normal_full.shape[0]
    normal = unpack_normal_opengl(normal_full[b:-b, b:-b] if b else normal_full)

    # Integrate in grid units, then convert to world units: the XY grid spans
    # [-1,1] over the FULL output width, so one pixel step is 2/size.
    h_world = integrate(normal) * (2.0 / size)

    # Anchor: the LR height carries every frequency the input actually has.
    lr = to_float(read(os.path.join(mat_dir, "SVBRDF", "height.png")))
    lb = b // (size // lr.shape[0])
    lr_crop = lr[lb:-lb, lb:-lb] if lb else lr
    side = normal.shape[0]
    anchor = cv2.resize(lr_crop, (side, side), interpolation=cv2.INTER_LINEAR)
    anchor_world = DISP_VAL * (anchor - 0.5)

    # High-pass the integrated height over the same band the anchor already
    # covers, so the two are not double counted.
    k = int(args.sigma * 4) | 1
    low = cv2.GaussianBlur(h_world, (k, k), args.sigma)
    combined = (h_world - low) + anchor_world

    # Back to the normalised storage the other height maps use.
    h_norm = combined / DISP_VAL + 0.5
    out = np.clip(h_norm, 0.0, 1.0)
    path = os.path.join(run_dir, "height_opt.png")
    if not cv2.imwrite(path, (out * U16_MAX).astype(np.uint16)):
        raise SystemExit("cv2.imwrite failed for " + path)

    hp = h_world - low
    print("  %-28s %s  %d^2   high-pass std %.2e   clipped %.2f%%"
          % (name, os.path.basename(run_dir), side, hp.std(),
             100.0 * np.mean((h_norm < 0) | (h_norm > 1))))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/runs/matsynth")
    ap.add_argument("--material")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--tile-width", type=int, default=17)
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="high-pass cutoff; 2 matches the x4 LR Nyquist")
    args = ap.parse_args()

    if args.material:
        dirs = [os.path.join(args.root, args.material)]
    elif args.all:
        dirs = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                      if os.path.isdir(os.path.join(d, "SVBRDF")))
    else:
        raise SystemExit("pass --material <name> or --all")

    print("reconstructing height for %d material(s)" % len(dirs))
    done = [d for d in dirs if process(d, args)]
    print("\n%d height map(s) written" % len(done))


if __name__ == "__main__":
    main()
