"""
LR renders and their S-upsamplings, cached per (material, S settings, light count).

File-level mirror of MatUp's first two stages, so the cache can be handed to
MatUp's own MatUpDataset as --upsamplings_dir unchanged:

    radiance_LR_L<n>/0000_LR_render.png ...        render_lowres.main_render, 16-bit
    radiance_<tag>_L<n>/render_0000_<name>.png ... S output, 8-bit, like sr_swinir.py

S reads the 16-bit renders with cv2.IMREAD_COLOR (-> 8-bit) and /255, exactly
as SwinIR's get_image_pair does for real_sr.

The manifest sits NEXT TO each directory (<dir>.json), never inside it:
MatUpDataset.load_renderings globs '*' in the directory and would try to
decode a JSON file as an image.

Experiment 1 and experiment 4 share this cache, which is the point: the
diffusion upsampler is by far the most expensive stage and runs once.
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch

from . import paths

paths.add_matup()


def lr_render_dir(root, n_lights):
    return os.path.join(root, "radiance_LR_L%d" % n_lights)


def sr_dir(root, tag, n_lights):
    return os.path.join(root, "radiance_%s_L%d" % (tag, n_lights))


def _complete(d, n):
    man = d.rstrip("/") + ".json"
    return os.path.isfile(man) and len(glob.glob(os.path.join(d, "*.png"))) == n


def _write_manifest(d, info):
    with open(d.rstrip("/") + ".json", "w") as fh:
        json.dump(info, fh, indent=2)


def ensure_lr_renders(lr_dir, root, n_lights=100, lr_size=128):
    """MatUp's own LR render (AA 2x2, Fibonacci lights x1.4, --ogl_normals)."""
    out = lr_render_dir(root, n_lights)
    if _complete(out, n_lights):
        return out
    from render_lowres import main_render
    args = argparse.Namespace(input_mat=lr_dir.rstrip("/") + "/", render_dir=out + "/",
                              crop_size=lr_size, AA_upscale=2, nb_lights=n_lights,
                              disp_val=0.01, light_falloff=False, opengl_normals=True)
    os.makedirs(out, exist_ok=True)
    main_render(args)
    n = len(glob.glob(os.path.join(out, "*.png")))
    if n != n_lights:
        # generate_fibonacci keeps the upper half of 2n sphere points, which is
        # n only for well-behaved n. MatUp itself would silently run with n.
        raise RuntimeError("expected %d LR renders, got %d in %s" % (n_lights, n, out))
    _write_manifest(out, {"n_lights": n_lights, "lr_size": lr_size, "AA_upscale": 2,
                          "source": lr_dir, "written": time.strftime("%Y-%m-%d %H:%M:%S")})
    return out


def read_lr_renders(d):
    """-> (N, 3, h, w) float tensor, RGB, 8-bit quantised exactly like SwinIR's reader."""
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    arr = [cv2.imread(f, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0 for f in files]
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous()


def read_sr_renders(d):
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    arr = [cv2.imread(f, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0 for f in files]
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous()


def ensure_sr_renders(lr_dir, root, upscaler, device, n_lights=100, lr_size=128):
    """Returns (lr_render_dir, sr_render_dir). Idempotent."""
    lr_out = ensure_lr_renders(lr_dir, root, n_lights, lr_size)
    out = sr_dir(root, upscaler.tag, n_lights)
    if _complete(out, n_lights):
        return lr_out, out
    x = read_lr_renders(lr_out).to(device)
    t0 = time.perf_counter()
    y = upscaler(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    os.makedirs(out, exist_ok=True)
    y8 = (y.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    for i, img in enumerate(y8):
        cv2.imwrite(os.path.join(out, "render_%04d_%s.png" % (i, upscaler.name)),
                    np.ascontiguousarray(img[:, :, ::-1]))
    info = dict(upscaler.config())
    info.update({"n_lights": n_lights, "seconds": secs, "source": lr_out,
                 "written": time.strftime("%Y-%m-%d %H:%M:%S")})
    _write_manifest(out, info)
    return lr_out, out
