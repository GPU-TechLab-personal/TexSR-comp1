"""
One MatUp run = (material set, material, S settings, variant). Shared by
experiment 1 (정밀: residuals after optimisation) and experiment 4 (variants),
so the "MatUp 원본" arm is computed once and read by both.

Result layout:
    /expdata/runs/matup/<set>/<S tag>/<material>/<variant>/
        result.json       metrics vs GT, history, timings, S config, extras
        maps/<ts>/...     MatUp's own 512^2 PNG outputs (8px border, DirectX normal)
        residual_mean.npy per-texel mean residual over lights (orig only, float16)

Variants:
    init   no optimisation: the bilinear-x4 initialisation (reference row)
    orig   MatUp as published (l2,1 + metallic reg), eager loop
    <tag>  orig + an extra loss term (extra_factory), e.g. exp4 "A_l0.01"

Metrics: claimed maps are basecolor, normal, roughness, metallic (MatUp has no
height output). The render-space numbers render MatUp's maps with the
bilinear LR height, which is what MatUp itself renders with.
"""
import json
import os
import time

import numpy as np
import torch

from variants.experiments import matup_driver as D
from variants.experiments.common import metrics as M
from variants.experiments.common import paths
from variants.experiments.common.protocol import load_material, crop_border
from variants.experiments.common.sr_cache import ensure_sr_renders

CLAIMED = ["basecolor", "normal", "roughness", "metallic"]
BORDER = 8                                   # (tile_width - 1) / 2


def run_dir(set_name, s_tag, material, variant):
    return os.path.join(paths.RUNS, "matup", set_name, s_tag, material, variant)


def load_result(set_name, s_tag, material, variant):
    p = os.path.join(run_dir(set_name, s_tag, material, variant), "result.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def run_one(set_name, material, upscaler, device, variant="orig", extra_factory=None,
            epochs=10, residuals=False, force=False, n_lights=100, render_lights=16,
            extra_info=None, log=print):
    mat_dir = os.path.join(paths.MATERIALS, set_name, material)
    out = run_dir(set_name, upscaler.tag, material, variant)
    res_path = os.path.join(out, "result.json")
    if os.path.isfile(res_path) and not force:
        log("  %-30s %-16s cached" % (material, variant))
        return json.load(open(res_path))

    t0 = time.perf_counter()
    cache = os.path.join(paths.SR_CACHE, set_name, material)
    lr_render, sr = ensure_sr_renders(os.path.join(mat_dir, "LR"), cache, upscaler, device, n_lights)
    t_sr = time.perf_counter() - t0
    args = D.matup_args(os.path.join(mat_dir, "LR"), lr_render, sr, os.path.join(out, "maps"),
                        nb_epochs=epochs, nb_lights=n_lights,
                        up_batch_size=10 if n_lights % 10 == 0 else 1)

    lr = load_material(os.path.join(mat_dir, "LR"))
    gt = {k: crop_border(v, BORDER) for k, v in load_material(os.path.join(mat_dir, "GT")).items()}
    result = {"set": set_name, "material": material, "variant": variant,
              "upscaler": upscaler.config(), "epochs": epochs, "n_lights": n_lights,
              "sr_cache": sr, "seconds_sr_stage": t_sr}
    if extra_info:
        result["extra_info"] = extra_info

    if variant == "init":
        maps, dataset = D.init_maps(args, device)
        history = []
    else:
        log("  %-30s %-16s optimising (%s)" % (material, variant, upscaler.tag))
        maps, _model, dataset, history = D.optimise(args, device, extra_factory, log=log)
        D.write_outputs(maps, args)
    result["history"] = history
    result["seconds_optimise"] = time.perf_counter() - t0 - t_sr

    pred = D.to_canonical(maps, lr_height=lr["height"], full_size=gt["basecolor"].shape[0] + 2 * BORDER)
    result["metrics"] = M.evaluate(pred, gt, device, maps=CLAIMED, border=0,
                                   n_lights=render_lights,
                                   full_size=gt["basecolor"].shape[0] + 2 * BORDER)
    result["metrics"]["height_note"] = "height = bilinear LR (MatUp outputs none); not a prediction"

    if residuals:
        stats, rmap = D.per_light_residuals(maps, dataset, args, device)
        result["residuals"] = stats
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "residual_mean.npy"), rmap)

    D.save_json(result, res_path)
    m = result["metrics"]["maps"]
    log("  %-30s %-16s bc PSNR %.2f  n %.2fdeg  r %.4f  FLIP %.4f  (%.0fs)" % (
        material, variant, m["basecolor"]["psnr"] or float("inf"), m["normal"]["angle_mean"],
        m["roughness"]["mae"], result["metrics"]["render"]["flip"], time.perf_counter() - t0))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
