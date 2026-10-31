#!/usr/bin/env python3
"""
실험 5: inference speed and memory, one protocol for every model.

    python variants/experiments/exp5_benchmark/bench_infer.py
    python variants/experiments/exp5_benchmark/bench_infer.py --models rrdbnet swinir --reps 10   # gate

Protocol (identical for every entry):
  * LR 128^2 -> x4 (512^2), batch 1, fp16 (model.half(), fp16 inputs)
  * 10 warm-up runs, then the mean of 100, each bracketed by
    torch.cuda.synchronize(); torch.cuda.max_memory_allocated() after
    reset_peak_memory_stats() - so peak memory includes activations
  * DiT: VAE encode of the conditioning + N denoiser steps + VAE decode, all
    inside the timed region (steps 1, 4, 20, 50)

Entries
  rrdbnet_1map     RRDBNet 23 blocks, one 3-channel map
  rrdbnet_5map     the same net over all 5 maps (independent input: 5 passes batched)
  rrdbnet_concat   15->15 channel RRDBNet, all maps in one pass
  swinir           SwinIR-L x4 (real_sr) on one image - MatUp calls it on 100 renders
  dit_s<N>         DiT-XL/2 latent denoiser (diffusers DiTTransformer2DModel) with the
                   LR latent concatenated as conditioning, SD-architecture VAE.
                   NO public SVBRDF DiT checkpoint exists, so weights are random:
                   time and memory depend on architecture and shapes, not on
                   weight values. Stated in the output, not hidden.
  matup_mlp        MatUp's per-material MLP (17x17 conv + 3x 1x1, 128 wide) over the
                   padded 528^2 7-channel init - one full-material evaluation

Output: /expdata/runs/exp5/infer_<gpu>.json and a markdown table.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.experiments.common import paths                        # noqa: E402
from variants.experiments.exp2_esrgan.core.model import RRDBNet, count_params   # noqa: E402

LR = 128


def timed(fn, warmup, reps):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    t = torch.tensor(times)
    return {"ms_mean": float(t.mean() * 1e3), "ms_std": float(t.std() * 1e3),
            "ms_min": float(t.min() * 1e3), "peak_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "reps": reps, "warmup": warmup}


def rrdb(dev, in_ch, batch):
    net = RRDBNet(in_ch, in_ch).to(dev).half().eval()
    x = torch.rand(batch, in_ch, LR, LR, device=dev, dtype=torch.half)
    return (lambda: net(x)), {"params_m": count_params(net) / 1e6}


def swinir(dev):
    """fp16 via autocast, not .half(): SwinIR builds its shifted-window attention
    mask in fp32 at every call (network_swinir.py calculate_mask), and a
    half() model then fails the attn @ v matmul on mixed dtypes."""
    from variants.experiments.common.upscalers import SwinIR
    s = SwinIR(dev)
    x = torch.rand(1, 3, LR, LR, device=dev)

    def run():
        with torch.autocast("cuda", dtype=torch.float16):
            return s.up(x)
    return run, {"params_m": count_params(s.model) / 1e6, "fp16": "autocast"}


def dit(dev, steps, size="XL"):
    from diffusers import AutoencoderKL, DiTTransformer2DModel
    cfg = {"XL": (28, 16, 72), "L": (24, 16, 64), "B": (12, 12, 64)}[size]
    tr = DiTTransformer2DModel(num_attention_heads=cfg[1], attention_head_dim=cfg[2], in_channels=8,
                               out_channels=8, num_layers=cfg[0], patch_size=2, sample_size=LR * 4 // 8,
                               num_embeds_ada_norm=1000).to(dev).half().eval()
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                        down_block_types=["DownEncoderBlock2D"] * 4, up_block_types=["UpDecoderBlock2D"] * 4,
                        block_out_channels=[128, 256, 512, 512], layers_per_block=2,
                        sample_size=512).to(dev).half().eval()
    lr = torch.rand(1, 3, LR, LR, device=dev, dtype=torch.half)
    cls = torch.zeros(1, dtype=torch.long, device=dev)

    def run():
        up = torch.nn.functional.interpolate(lr, scale_factor=4, mode="bicubic")
        cond = vae.encode(up * 2 - 1).latent_dist.mode()
        z = torch.randn_like(cond)
        for i in range(steps):
            t = torch.full((1,), 999 - i * (1000 // steps), device=dev)
            eps = tr(torch.cat([z, cond], 1), timestep=t, class_labels=cls).sample[:, :4]
            z = z - eps / steps                               # scheduler arithmetic is negligible
        return vae.decode(z).sample

    return run, {"params_m": count_params(tr) / 1e6, "vae_params_m": count_params(vae) / 1e6,
                 "dit_size": size, "steps": steps, "weights": "random (architecture timing only)"}


def matup_mlp(dev):
    paths.add_matup()
    from matup.model import FullyConnected
    m = FullyConnected(7, 7, 17, 4, 128, "LeakyReLU").to(dev).half().eval()
    x = torch.rand(1, 7, LR * 4 + 16, LR * 4 + 16, device=dev, dtype=torch.half)
    return (lambda: m(x)), {"params_m": count_params(m) / 1e6}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    default=["rrdbnet_1map", "rrdbnet_5map", "rrdbnet_concat", "swinir", "dit", "matup_mlp"])
    ap.add_argument("--dit-steps", nargs="+", type=int, default=[1, 4, 20, 50])
    ap.add_argument("--dit-size", default="XL", choices=["XL", "L", "B"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(paths.RUNS, "exp5"))
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    gpu = torch.cuda.get_device_name(0)

    jobs = []
    for m in args.models:
        if m == "rrdbnet_1map":
            jobs.append((m, lambda: rrdb(dev, 3, 1)))
        elif m == "rrdbnet_5map":
            jobs.append((m, lambda: rrdb(dev, 3, 5)))
        elif m == "rrdbnet_concat":
            jobs.append((m, lambda: rrdb(dev, 15, 1)))
        elif m == "swinir":
            jobs.append((m, lambda: swinir(dev)))
        elif m == "dit":
            for s in args.dit_steps:
                jobs.append(("dit_s%d" % s, lambda s=s: dit(dev, s, args.dit_size)))
        elif m == "matup_mlp":
            jobs.append((m, lambda: matup_mlp(dev)))
        else:
            raise SystemExit("unknown model " + m)

    res = {"gpu": gpu, "torch": torch.__version__, "lr": LR, "scale": 4, "dtype": "fp16", "entries": {}}
    for name, build in jobs:
        torch.cuda.empty_cache()
        with torch.no_grad():
            fn, info = build()
            r = timed(fn, args.warmup, args.reps)
        r.update(info)
        res["entries"][name] = r
        print("  %-16s %9.2f ms  +-%6.2f   peak %7.0f MB   %s" % (
            name, r["ms_mean"], r["ms_std"], r["peak_mem_mb"],
            ", ".join("%s=%s" % kv for kv in info.items())), flush=True)
        del fn
    os.makedirs(args.out, exist_ok=True)
    tag = gpu.replace(" ", "_")
    json.dump(res, open(os.path.join(args.out, "infer_%s.json" % tag), "w"), indent=2)
    md = ["| model | ms (mean±std) | peak MB | params (M) | note |", "|---|---|---|---|---|"]
    for n, r in res["entries"].items():
        md.append("| %s | %.2f ± %.2f | %.0f | %.2f | %s |" % (n, r["ms_mean"], r["ms_std"], r["peak_mem_mb"],
                                                             r["params_m"], r.get("weights", "")))
    open(os.path.join(args.out, "infer_%s.md" % tag), "w").write(
        "# inference, %s, LR %d^2 x4, fp16, batch 1\n\n" % (gpu, LR) + "\n".join(md) + "\n")
    print("-> " + args.out)


if __name__ == "__main__":
    main()
