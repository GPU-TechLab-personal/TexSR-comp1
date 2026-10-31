"""What leaves the driver: MatUp's PNGs (change (e)) and the canonical material."""
import json
import os

import numpy as np
import torch

from variants.experiments.matup_driver.upstream import data_utils, render_utils


def write_outputs(maps_eval, args):
    """MatUp's own 512^2 PNG outputs (8px zero border, DirectX normal)."""
    # (e) straight into args.output_folder: main_optim.py:51-52 appends a
    # timestamp folder first
    os.makedirs(args.output_folder, exist_ok=True)
    data_utils.output_maps_from_list([m.clone() for m in maps_eval], args, "_opt", "",
                                     pad_output=True)


def to_canonical(maps_eval, lr_height=None, full_size=512):
    """MatUp's internal maps -> canonical material (protocol.py), valid region only.

    MatUp does not output height. If `lr_height` (LR canonical height) is given,
    the height slot is filled with its bilinear x4 over the same region - what
    MatUp itself renders with (dataLoader.py:115-128) - and must be reported as
    such, never as a MatUp prediction."""
    import cv2
    albedo, metallic, normal, alpha = [m.detach().float().cpu() for m in maps_eval[:4]]
    out = {
        "basecolor": render_utils.gammaCorrection(albedo.clamp(0, 1)).numpy(),
        "metallic": metallic[..., 0].clamp(0, 1).numpy(),
        "roughness": torch.sqrt(alpha[..., 0].clamp(0, 1)).numpy(),
    }
    n = data_utils.torch_norm(normal, dimToNorm=-1).numpy().copy()
    n[..., 1] = -n[..., 1]                      # internal -> OpenGL
    out["normal"] = n
    if lr_height is not None:
        size = out["basecolor"].shape[0]
        b = (full_size - size) // 2
        h = cv2.resize(lr_height, (full_size, full_size), interpolation=cv2.INTER_LINEAR)
        out["height"] = h[b:full_size - b, b:full_size - b]
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
