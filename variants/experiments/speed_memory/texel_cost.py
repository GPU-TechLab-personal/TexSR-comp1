#!/usr/bin/env python3
"""
실험 5 (압축 관점): cost of restoring one output texel, ESRGAN-style vs
MatUp-style (per-material MLP).

    python variants/experiments/speed_memory/texel_cost.py

Counted, not estimated: multiply-accumulates are summed by forward hooks on
every Conv2d / Linear over a real forward pass, then divided by the number
of output texels that pass produced (per map where the model is per map).

  esrgan (independent)   one RRDBNet pass per map: MACs / (4h * 4w) per map-texel;
                         x5 for a full material texel
  esrgan (concat)        one 15-ch pass gives all 5 maps: MACs / (4h * 4w)
  matup MLP              17x17x7 -> 128 -> 128 -> 128 -> 7 per output texel
                         (all 7 data channels at once), plus the per-material
                         optimisation that must precede it (amortised line,
                         from bench_train.py if its JSON exists)

Storage per material (what has to ship to reconstruct 512^2):
  esrgan   the 128^2 LR maps; the network is shared by every material
  matup    the 128^2 LR maps + that material's MLP weights (fp16)
  raw      the 512^2 maps themselves (16-bit)
"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.experiments.common import paths                          # noqa: E402
from variants.experiments.esrgan_map_degradation.core.model import RRDBNet, count_params  # noqa: E402

CH = {"basecolor": 3, "normal": 3, "roughness": 1, "metallic": 1, "height": 1}


def macs(model, x):
    total = [0]

    def hook(m, inp, out):
        if isinstance(m, torch.nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
            total[0] += out.numel() * k
        elif isinstance(m, torch.nn.Linear):
            total[0] += out.numel() * m.in_features

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return total[0]


def main():
    lr = 128
    out_texels = (4 * lr) ** 2
    rows = {}
    r3 = RRDBNet(3, 3)
    m3 = macs(r3, torch.zeros(1, 3, lr, lr))
    rows["esrgan_independent"] = {"macs_per_map_texel": m3 / out_texels,
                                  "macs_per_material_texel": 5 * m3 / out_texels,
                                  "params": count_params(r3), "params_shared_across_materials": True}
    r15 = RRDBNet(15, 15)
    m15 = macs(r15, torch.zeros(1, 15, lr, lr))
    rows["esrgan_concat"] = {"macs_per_material_texel": m15 / out_texels, "params": count_params(r15),
                             "params_shared_across_materials": True}
    paths.add_matup()
    from matup.model import FullyConnected
    mlp = FullyConnected(7, 7, 17, 4, 128, "LeakyReLU")
    mm = macs(mlp, torch.zeros(1, 7, 17, 17))
    rows["matup_mlp"] = {"macs_per_material_texel": mm, "params": count_params(mlp),
                         "params_shared_across_materials": False,
                         "note": "4 maps (no height); MLP must first be optimised per material"}
    bt = sorted(glob.glob(os.path.join(paths.RUNS, "exp5", "train_*.json")))
    if bt:
        e = json.load(open(bt[-1]))["entries"].get("matup")
        if e:
            rows["matup_mlp"]["optimisation_seconds_per_material"] = e["seconds_per_material_10ep_eager"]

    lr_bytes = sum(CH.values()) * lr * lr * 2
    hr_bytes = sum(CH.values()) * out_texels * 2
    storage = {"raw_hr_16bit": hr_bytes, "esrgan_lr_16bit": lr_bytes,
               "matup_lr_16bit_plus_mlp_fp16": lr_bytes + 2 * rows["matup_mlp"]["params"]}

    print("| method | MACs / output texel (all maps) | params | shared? |")
    print("|---|---|---|---|")
    for k, r in rows.items():
        print("| %s | %.3g | %.2fM | %s |" % (k, r["macs_per_material_texel"], r["params"] / 1e6,
                                             r["params_shared_across_materials"]))
    print("\nstorage per 512^2 material: " + ", ".join("%s %.1f KB" % (k, v / 1024) for k, v in storage.items()))
    out = os.path.join(paths.RUNS, "exp5", "texel_cost.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"per_texel": rows, "storage_bytes": storage, "lr": lr, "scale": 4}, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
