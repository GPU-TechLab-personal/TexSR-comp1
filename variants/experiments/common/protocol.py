"""
Canonical material representation and the unified LR protocol (공통 평가 프로토콜 / 공정성).

Canonical in-memory material: a dict of float32 numpy arrays
    basecolor  HxWx3  sRGB-encoded, [0,1], RGB order
    normal     HxWx3  unit vectors, OpenGL convention (+Y up, MatSynth's)
    roughness  HxW    perceptual roughness r in [0,1]; the GGX alpha is r**2
                      (MatUp's convention: dataLoader.py:114)
    metallic   HxW    [0,1]
    height     HxW    [0,1]

On disk it is MatUp's layout (so a folder can be fed to MatUp as --input_mat
with --ogl_normals): baseColor/normal/roughness/metallic/height .png, all
16-bit, colour files in cv2's BGR byte order.

LR protocol - one function, used for every method, every experiment:
    basecolor  area-average in LINEAR light, then back to sRGB
    normal     area-average the unit vectors, then renormalise
    roughness  NDF-variance preserving. The block's normal spread is folded
               into alpha^2 instead of being lost when the normals are averaged:
                 lean     alpha'^2 = mean(alpha^2) + Var(slope_x) + Var(slope_y)
                 toksvig  alpha'^2 = mean(alpha^2) + 2 (1-|n_mean|) / |n_mean|
               (Beckmann slope-space variance, per-axis alpha^2/2; to first
               order in the tilt the two agree.) `area` = plain mean of r, for
               comparison only.
    metallic, height   area-average
The factor must be an integer; blocks are exact (no resampling kernel).
"""
import json
import os

import cv2
import numpy as np

MAP_FILES = {
    "basecolor": "baseColor.png",
    "normal": "normal.png",
    "roughness": "roughness.png",
    "metallic": "metallic.png",
    "height": "height.png",
}
MAP_NAMES = list(MAP_FILES)
COLOR_MAPS = ("basecolor", "normal")
ROUGHNESS_MODES = ("lean", "toksvig", "area")

# -------------------------------------------------------------- colour space

def srgb_to_linear(x):
    """Same formula as MatUp's file_utils.srgb2linrgb."""
    x = np.asarray(x, np.float64)
    return np.where(x > 0.04045, ((np.maximum(x, 0.04045) + 0.055) / 1.055) ** 2.4, x / 12.92)


def linear_to_srgb(x):
    """Same formula as MatUp's render_utils.gammaCorrection."""
    x = np.asarray(x, np.float64)
    return np.where(x > 0.0031308, 1.055 * np.maximum(x, 0.0031308) ** (1 / 2.4) - 0.055, 12.92 * x)


# ------------------------------------------------------------------- helpers

def normalize(n, eps=1e-12):
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), eps)


def block_mean(a, f):
    h, w = a.shape[:2]
    if h % f or w % f:
        raise ValueError("size %dx%d not divisible by %d" % (w, h, f))
    a = np.asarray(a, np.float64)
    return a.reshape(h // f, f, w // f, f, *a.shape[2:]).mean(axis=(1, 3))


def block_var(a, f):
    m = block_mean(a, f)
    return block_mean(np.asarray(a, np.float64) ** 2, f) - m ** 2


def to_float(arr):
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def center_crop(a, size):
    h, w = a.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return a[y:y + size, x:x + size]


def crop_border(a, b):
    return a[b:-b, b:-b] if b else a


# ----------------------------------------------------------- decode / encode

def from_raw(raw):
    """dict of decoded cv2 arrays (any depth, BGR) -> canonical material.
    Any subset of the five maps."""
    def plane(a):
        return a if a.ndim == 2 else a[:, :, 0]

    def rgb(a):
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if a.ndim == 2 else a[:, :, :3]
        return a[:, :, ::-1]

    conv = {"basecolor": lambda a: to_float(rgb(a)),
            "normal": lambda a: normalize(to_float(rgb(a)) * 2.0 - 1.0).astype(np.float32),
            "roughness": lambda a: to_float(plane(a)),
            "metallic": lambda a: to_float(plane(a)),
            "height": lambda a: to_float(plane(a))}
    return {k: np.ascontiguousarray(conv[k](v), dtype=np.float32) for k, v in raw.items()}


def _u16(x):
    return np.round(np.clip(x, 0.0, 1.0) * 65535.0).astype(np.uint16)


def save_material(mat, folder):
    os.makedirs(folder, exist_ok=True)
    for name, fn in MAP_FILES.items():
        a = mat[name]
        if name == "normal":
            a = 0.5 * normalize(a) + 0.5
        a = _u16(a)
        if a.ndim == 3:
            a = a[:, :, ::-1]                       # RGB -> BGR on disk
        if not cv2.imwrite(os.path.join(folder, fn), np.ascontiguousarray(a)):
            raise RuntimeError("cv2.imwrite failed: " + os.path.join(folder, fn))


def load_material(folder, suffix=""):
    raw = {}
    for name, fn in MAP_FILES.items():
        path = os.path.join(folder, fn.replace(".png", suffix + ".png"))
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise FileNotFoundError(path)
        raw[name] = a
    return from_raw(raw)


def load_meta(mat_dir):
    with open(os.path.join(mat_dir, "meta.json")) as fh:
        return json.load(fh)


# ------------------------------------------------------------- the protocol

def downsample(mat, f, roughness_mode="lean"):
    """Integer-factor LR generation. See the module docstring.

    Works on any subset of the maps (the shard readers stream one column at a
    time to stay inside memory), with one coupling: roughness under lean /
    toksvig needs the full-resolution normal map in the same call."""
    if roughness_mode not in ROUGHNESS_MODES:
        raise ValueError(roughness_mode)
    if f == 1:
        return {k: v.copy() for k, v in mat.items()}
    if "roughness" in mat and roughness_mode != "area" and "normal" not in mat:
        raise ValueError("roughness_mode=%s needs the normal map alongside roughness" % roughness_mode)

    out = {}
    if "basecolor" in mat:
        out["basecolor"] = linear_to_srgb(block_mean(srgb_to_linear(mat["basecolor"]), f))

    if "normal" in mat:
        n = normalize(np.asarray(mat["normal"], np.float64))
        n_mean = block_mean(n, f)
        length = np.linalg.norm(n_mean, axis=-1)
        out["normal"] = normalize(n_mean)

    if "roughness" in mat:
        r = np.asarray(mat["roughness"], np.float64)
        if roughness_mode == "area":
            out["roughness"] = block_mean(r, f)
        else:
            a2 = block_mean(r ** 4, f)                    # alpha^2 = r^4
            if roughness_mode == "lean":
                nz = np.maximum(n[..., 2], 1e-3)
                a2 = a2 + block_var(-n[..., 0] / nz, f) + block_var(-n[..., 1] / nz, f)
            else:
                L = np.clip(length, 1e-6, 1.0)
                a2 = a2 + 2.0 * (1.0 - L) / L
            out["roughness"] = np.clip(a2, 0.0, 1.0) ** 0.25

    for k in ("metallic", "height"):
        if k in mat:
            out[k] = block_mean(mat[k], f)
    return {k: np.ascontiguousarray(np.clip(v, -1 if k == "normal" else 0, 1), dtype=np.float32)
            for k, v in out.items()}


def make_pair(src, gt_size, scale=4, mode="resize", roughness_mode="lean"):
    """Source (e.g. 4096^2) -> (GT, LR). LR is ALWAYS the protocol applied to GT.

    resize: GT = protocol(src, src/gt_size)   (whole material, band-limited)
    crop:   GT = native-resolution centre window of gt_size
    """
    s = next(iter(src.values())).shape[0]
    if mode == "resize":
        if s % gt_size:
            raise ValueError("source %d not an integer multiple of GT %d" % (s, gt_size))
        gt = downsample(src, s // gt_size, roughness_mode)
    elif mode == "crop":
        gt = {k: np.ascontiguousarray(center_crop(v, gt_size)) for k, v in src.items()}
    else:
        raise ValueError(mode)
    lr = downsample(gt, scale, roughness_mode)
    return gt, lr


# ------------------------------------------------ geometric augmentation

def flip_rot(mat, k, flip):
    """rot90 by k and optional horizontal flip, with the normal vectors
    transformed along with the pixels (a flipped normal map that keeps its
    old X component is lit from the wrong side).

    Image axes: x = column (right), y = row (down). OpenGL normals have +Y up,
    i.e. opposite to the row axis.
    """
    out = {}
    for name, a in mat.items():
        a = np.rot90(a, k, axes=(0, 1))
        if flip:
            a = a[:, ::-1]
        out[name] = a
    n = out["normal"].copy()
    for _ in range(k % 4):
        # np.rot90 k=1 turns the image counter-clockwise; a vector drawn on it
        # turns CCW too: (x, y_up) -> (-y_up, x).
        n = np.stack([-n[..., 1], n[..., 0], n[..., 2]], axis=-1)
    if flip:
        n[..., 0] = -n[..., 0]
    out["normal"] = n
    return {k_: np.ascontiguousarray(v, dtype=np.float32) for k_, v in out.items()}
