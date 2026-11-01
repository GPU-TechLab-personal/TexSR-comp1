"""Write an SR arm's evaluation folder - the input every evaluator reads.

Any super-resolution method (DeepTextures in TF, ESRGAN / SwinIR in torch, ...)
becomes an arm by writing this folder; the map report, render eval, perceptual
eval and the DT-only montages then take it as `--arm TAG=DIR`, and
variants/arms.json gives the tag its name and comparisons.

THE CONTRACT

    <DIR>/<material>/<map>.png         for the maps the arm produces, any subset of
                                       baseColor, normal, roughness, metallic, height
                                       (+ render: the image-SR of MatUp's render_lr.png)
        size     the GT's (512^2 for a x4 arm on the 128^2 MatSynth LR)
        normal   OpenGL (+Y up), 8-bit RGB
        scalar   1-channel 8-bit (roughness, metallic, height)
        colour   3-channel 8-bit (baseColor, normal, render)
    <DIR>/<material>/sr_manifest.json
        material, checkpoint, step, sub_name, scale, matup_run, scalar_collapse,
        collapse_weights, outputs.<map>{src, kind, source_encoding, lr_shape,
        sr_shape[, channel_spread]}
        + any provenance keys the method adds (checkpoint_file, weight_sha, ...).
        checkpoint_file, when present, is the file Get-CompleteEval (util/lib.ps1)
        checks is still there and older than the manifest; without it the
        checkpoint is a TF prefix and <checkpoint>.index is checked.

The render eval needs all five maps at the GT size (svbrdf_render_eval dies
otherwise); the map report takes any subset and fits sizes itself.

WHAT IS HERE (numpy + cv2 only, so any image can write an arm)

    load_as_rgb01   any map encoding -> HxWx3 float32 [0,1] (map_encode.to_rgb8, RGB)
    save_colour / save_sr   a 3-channel SR output back to the map's PNG (scalar maps:
                    collapse the three channels with COLLAPSE[...], record the spread)
    tile_grid / super_resolve(predict, ...)   tiled inference around any
                    predict(batch NHWC float32) -> NHWC float32
    write_arm       the per-material loop: inputs from the material and its newest
                    MatUp run, predict, save, sr_manifest.json
"""
import glob
import json
import os

import cv2
import numpy as np

from variants.util import map_encode
from variants.matup.baseline.matup_outputs import newest_run

# name -> (source, file, kind). "svbrdf" sources come from the material's LR set,
# "run" sources from the MatUp output folder. kind decides how the map is squeezed
# into and back out of a 3-channel network.
ALL_JOBS = [
    ("baseColor", "svbrdf", "baseColor.png", "colour"),
    ("normal",    "svbrdf", "normal.png",    "colour"),
    ("roughness", "svbrdf", "roughness.png", "scalar"),
    ("metallic",  "svbrdf", "metallic.png",  "scalar"),
    ("height",    "svbrdf", "height.png",    "scalar"),
    ("render",    "run",    "render_lr.png", "colour"),
]

COLLAPSE = {
    # symmetric - matches the fact that the three input channels were identical
    "mean":  (1 / 3.0, 1 / 3.0, 1 / 3.0),
    # correct BT.601 luma on RGB
    "luma":  (0.299, 0.587, 0.114),
    # what DeepTextures' test.py metric path actually applies, R/B swapped
    "paper": (0.114, 0.587, 0.299),
}


def load_as_rgb01(path):
    """Any of the five map encodings -> HxWx3 float32 in [0,1], plus provenance.

    One cv2 path for everything, because the inputs are genuinely heterogeneous:
    8-bit RGB (baseColor, normal), 8-bit grey (roughness, metallic), 16-bit grey
    (height) and 16-bit RGB (render_lr).

    cv2 hands back BGR and SR networks are trained on RGB, so the channel reversal
    is explicit and load-bearing (map_encode.to_rgb8 does it). For the grey maps it
    is a no-op because all three channels are equal. The same encoding builds the
    MatSynth training pairs (deeptex/matsynth/matsynth_make_pairs.py), so train and
    eval cannot drift.
    """
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError("cv2.imread returned None: %s" % path)
    src = {"dtype": str(arr.dtype),
           "channels": 1 if arr.ndim == 2 else int(arr.shape[2])}
    try:
        rgb = map_encode.to_rgb8(arr)
    except ValueError as e:
        raise RuntimeError("%s in %s" % (e, path))
    return rgb.astype(np.float32) / 255.0, src


def save_colour(path, sr):
    """A 3-channel map back to a 3-channel PNG. Returns (shape, None): no spread."""
    u8 = np.rint(np.clip(sr, 0.0, 1.0) * 255.0).astype(np.uint8)
    out = u8[:, :, ::-1]          # RGB -> BGR for cv2
    if not cv2.imwrite(path, out, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError("cv2.imwrite failed for " + path)
    return list(out.shape), None


def save_sr(path, sr, kind, collapse="mean"):
    """Write the super-resolved map back in the shape its column expects.

    Returns (shape, channel_spread) where channel_spread measures how far apart
    the three output channels actually were - if it is ~0 the collapse choice is
    immaterial, and if it is large it has to be stated.
    """
    if kind != "scalar":
        return save_colour(path, sr)
    f = sr.astype(np.float64)
    d = f.max(axis=2) - f.min(axis=2)
    spread = {"mean_255": float(d.mean() * 255.0),
              "p99_255": float(np.percentile(d, 99) * 255.0),
              "max_255": float(d.max() * 255.0)}
    w = COLLAPSE[collapse]
    out = np.rint(np.clip(f[:, :, 0] * w[0] + f[:, :, 1] * w[1] + f[:, :, 2] * w[2],
                          0.0, 1.0) * 255.0).astype(np.uint8)
    if not cv2.imwrite(path, out, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError("cv2.imwrite failed for " + path)
    return list(out.shape), spread


def select(all_jobs, maps):
    """The jobs named in `maps` ("all" or comma-separated); [] when none match."""
    wanted = None if maps == "all" else set(maps.split(","))
    return [j for j in all_jobs if wanted is None or j[0] in wanted]


def tile_grid(extent, tile, overlap):
    """Start offsets covering [0, extent) with the last one anchored to the edge."""
    if tile <= 0 or extent <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, extent - tile + 1, stride))
    if starts[-1] + tile < extent:
        starts.append(extent - tile)
    return starts


def super_resolve(predict, lr, scale, tile, overlap):
    """lr: HxWx3 float32 in [0,1]; predict(NHWC) -> NHWC. Returns (H*scale)x(W*scale)x3.

    tile > 0 runs overlapping tiles and discards half the overlap on each interior
    edge rather than blending: averaging two differently-conditioned outputs makes
    its own seam.
    """
    h, w = lr.shape[:2]
    if tile <= 0 or (h <= tile and w <= tile):
        out = predict(lr[None])[0]
        assert out.shape[0] == h * scale and out.shape[1] == w * scale, out.shape
        return out

    canvas = np.zeros((h * scale, w * scale, lr.shape[2]), np.float32)
    for y in tile_grid(h, tile, overlap):
        for x in tile_grid(w, tile, overlap):
            patch = lr[y:y + tile, x:x + tile]
            sr = predict(patch[None])[0]
            ty0 = 0 if y == 0 else overlap // 2
            tx0 = 0 if x == 0 else overlap // 2
            ty1 = patch.shape[0] if y + tile >= h else patch.shape[0] - overlap // 2
            tx1 = patch.shape[1] if x + tile >= w else patch.shape[1] - overlap // 2
            canvas[(y + ty0) * scale:(y + ty1) * scale,
                   (x + tx0) * scale:(x + tx1) * scale] = \
                sr[ty0 * scale:ty1 * scale, tx0 * scale:tx1 * scale]
    return canvas


def materials(root, material=""):
    """[<root>/<material>] or every folder under root with an SVBRDF/."""
    if material:
        return [os.path.join(root, material)]
    return sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.isdir(os.path.join(d, "SVBRDF")))


def write_arm(mats, out, jobs, sr_fn, save, collapse, provenance, run="", die=None):
    """The per-material loop every arm writer shares.

    mats        material folders (materials())
    out         the arm folder
    jobs        [(map, source, file, kind)] (ALL_JOBS or a subset)
    sr_fn(lr, where)  HxWx3 float32 [0,1] -> SR float32; `where` = "material/map"
                for the method's own error messages (e.g. OOM -> exit 10)
    save(path, sr, kind) -> (shape, channel_spread or None)
    collapse    (name, weights) the manifest records
    provenance  dict written first into every manifest: checkpoint, step,
                sub_name, scale (+ checkpoint_file, weight_sha, ... if the method has them)
    die(code, msg)  called on a non-finite output (exit 4)

    Returns (done material names, skipped "material/map (reason)" strings).
    """
    os.makedirs(out, exist_ok=True)
    done, skipped = [], []
    for mat_dir in mats:
        name = os.path.basename(mat_dir.rstrip("/"))
        run_dir = newest_run(mat_dir, name, run)
        out_dir = os.path.join(out, name)
        os.makedirs(out_dir, exist_ok=True)
        wrote = {}

        for key, where, fname, kind in jobs:
            if where == "svbrdf":
                src = os.path.join(mat_dir, "SVBRDF", fname)
            elif run_dir:
                src = os.path.join(run_dir, fname)
            else:
                skipped.append("%s/%s (no MatUp run)" % (name, key))
                continue
            if not os.path.exists(src):
                skipped.append("%s/%s (%s missing)" % (name, key, os.path.basename(src)))
                continue
            lr, prov = load_as_rgb01(src)
            sr = sr_fn(lr, "%s/%s" % (name, key))
            if not np.isfinite(sr).all():
                (die or _die)(4, "non-finite output on %s/%s" % (name, key))
            out_shape, spread = save(os.path.join(out_dir, key + ".png"), sr, kind)
            wrote[key] = {"src": src, "kind": kind, "source_encoding": prov,
                          "lr_shape": list(lr.shape), "sr_shape": out_shape}
            if spread:
                wrote[key]["channel_spread"] = spread

        manifest = {"material": name}
        manifest.update(provenance)
        manifest.update({"matup_run": os.path.basename(run_dir) if run_dir else None,
                         "scalar_collapse": collapse[0],
                         "collapse_weights": collapse[1],
                         "outputs": wrote})
        with open(os.path.join(out_dir, "sr_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        done.append(name)
        sp = [(k, v["channel_spread"]["mean_255"]) for k, v in sorted(wrote.items())
              if "channel_spread" in v]
        extra = ("   ch-spread " + " ".join("%s %.1f" % kv for kv in sp)) if sp else ""
        print("  %-28s %s%s" % (name, ", ".join(sorted(wrote)) or "nothing", extra))
    return done, skipped


def _die(code, msg):
    import sys
    print("[fatal] %s" % msg)
    sys.exit(code)
