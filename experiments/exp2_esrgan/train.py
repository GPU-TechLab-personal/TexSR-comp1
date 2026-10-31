#!/usr/bin/env python3
"""
Multi-map ESRGAN trainer for experiments 2 (ablations) and 3 (rendering loss).

    docker compose run --rm exp python experiments/exp2_esrgan/train.py \
        --config experiments/exp2_esrgan/configs/2-0_base_scratch.yaml
    # smoke / override any key:
    ... --config ... --set train.niter=20 eval.every=10 --run-name _smoke

One config = one arm. Every arm shares: data (MatSynth train split,
build_train_set.py), the LR protocol, the number of material patches per
iteration, the schedule and the evaluation, so an arm differs from the
baseline only in the factor it names.

Run directory: /expdata/runs/exp2/<run-name>/
    config.yaml          the resolved config
    log.jsonl            every log.every its: losses, it/s, peak memory, train_seconds
    eval.jsonl           every eval.every its: eval12 per-map metrics (fast subset)
    ckpt/iter_<n>.pt     G, D, optimisers, schedulers (keeps save.keep newest)
Resumes from the newest checkpoint automatically.

`train_seconds` counts only training steps (CUDA-synchronised at each log
point), not evaluation or checkpointing: experiment 5 turns it into GPU-hours
to reach a target metric (experiments/exp5_benchmark/gpu_hours.py).
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, "/workspace")
from experiments.common import paths                                  # noqa: E402
from experiments.common import metrics as M                           # noqa: E402
from experiments.common.protocol import load_material                 # noqa: E402
from experiments.exp2_esrgan import data as D                         # noqa: E402
from experiments.exp2_esrgan import losses as L                       # noqa: E402
from experiments.exp2_esrgan.model import (MAPS, MultiMapSR, VGGPerceptual,   # noqa: E402
                                           VGGStyleDiscriminator, count_params,
                                           ensure_pretrained, load_pretrained)


def load_config(path, overrides):
    cfg = yaml.safe_load(open(path))
    base = cfg.pop("base", None)
    if base:
        parent = load_config(os.path.join(os.path.dirname(path), base), [])
        cfg = deep_merge(parent, cfg)
    for o in overrides or []:
        k, v = o.split("=", 1)
        d = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            d = d.setdefault(kk, {})
        d[keys[-1]] = yaml.safe_load(v)
    return cfg


# Below these paths a child config REPLACES the parent's entry instead of
# merging into it: arm 2-3's `normal: {angular: 1.0}` means "angular only",
# and a recursive merge would silently keep the base's l1/vgg/gan beside it.
REPLACE_CHILDREN = {("loss", "per_map")}


def deep_merge(a, b, path=()):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if path in REPLACE_CHILDREN:
            out[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v, path + (k,))
        else:
            out[k] = v
    return out


def infinite(loader):
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def quick_eval(G, eval_set, device, amp):
    """eval12, full image, per-map primary metrics only (no LPIPS/FLIP - fast)."""
    G.eval()
    rows = []
    for name in eval_set:
        mdir = os.path.join(paths.MATERIALS, "eval12", name)
        lr = D.encode_np(load_material(os.path.join(mdir, "LR")))
        gt = load_material(os.path.join(mdir, "GT"))
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            sr = G(D.to_batch([lr], device))
        pred = D.decode_np({m: v[0] for m, v in sr.items()})
        bc = M.texsr_metrics.psnr(M._u16(pred["basecolor"]), M._u16(gt["basecolor"]))
        rows.append({"basecolor_psnr": bc if np.isfinite(bc) else np.nan,
                     "normal_angle": M.normal_metrics(pred["normal"], gt["normal"])["angle_mean"],
                     "roughness_mae": M.roughness_metrics(pred["roughness"], gt["roughness"])["mae"],
                     "metallic_mae": M.metallic_metrics(pred["metallic"], gt["metallic"])["mae"],
                     "height_mae": M.height_metrics(pred["height"], gt["height"])["mae"]})
    G.train()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (yaml values)")
    ap.add_argument("--run-name", help="default: the config's name")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    name = args.run_name or cfg["name"]
    run = os.path.join(paths.RUNS, cfg.get("experiment", "exp2"), name)
    os.makedirs(os.path.join(run, "ckpt"), exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(run, "config.yaml"), "w"), sort_keys=False)
    dev = torch.device("cuda:0")
    torch.manual_seed(cfg["train"].get("seed", 0))
    np.random.seed(cfg["train"].get("seed", 0))
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[cfg["train"].get("amp", "bf16")]
    tr = cfg["train"]

    # ---------------- model
    mc = cfg["model"]
    G = MultiMapSR(mc["arch"], mc["input"], mc.get("num_block", 23), mc.get("num_feat", 64),
                   mc.get("num_grow_ch", 32)).to(dev)
    init_report = {"mode": cfg["init"]["mode"]}
    if cfg["init"]["mode"] == "pretrained":
        path = ensure_pretrained(cfg["init"]["weights"], paths.PRETRAINED)
        init_report.update(load_pretrained(G, path))
    elif cfg["init"]["mode"] != "scratch":
        raise ValueError(cfg["init"]["mode"])

    per_map = cfg["loss"]["per_map"]
    gan_maps = [m for m in MAPS if per_map.get(m, {}).get("gan", 0) > 0]
    dmode = cfg["disc"]["mode"]
    discs = L.Discriminators(dmode, gan_maps, mc["arch"] == "separate",
                             lambda c: VGGStyleDiscriminator(c, cfg["disc"].get("num_feat", 64))).to(dev)
    need_vgg = any(v.get("vgg", 0) > 0 for v in per_map.values())
    crit = L.Criterion(cfg["loss"], dev, VGGPerceptual().to(dev) if need_vgg else None)
    if cfg["loss"].get("render", {}).get("weight", 0) > 0:
        crit.render = L.RenderLoss(cfg["loss"]["render"], dev, cfg["data"]["hr_patch"])

    opt_g = torch.optim.Adam(G.parameters(), lr=tr["lr_g"], betas=(0.9, 0.99))
    sch_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, tr["milestones"], tr.get("gamma", 0.5))
    opt_d = sch_d = None
    if discs.active():
        opt_d = torch.optim.Adam(discs.parameters(), lr=tr["lr_d"], betas=(0.9, 0.99))
        sch_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, tr["milestones"], tr.get("gamma", 0.5))

    start, train_seconds = 0, 0.0
    ckpts = sorted(glob.glob(os.path.join(run, "ckpt", "iter_*.pt")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if ckpts:
        st = torch.load(ckpts[-1], map_location=dev)
        G.load_state_dict(st["G"])
        opt_g.load_state_dict(st["opt_g"])
        sch_g.load_state_dict(st["sch_g"])
        if opt_d is not None and st.get("D") is not None:
            discs.load_state_dict(st["D"])
            opt_d.load_state_dict(st["opt_d"])
            sch_d.load_state_dict(st["sch_d"])
        start, train_seconds = st["iter"], st["train_seconds"]
        print("resumed at iter %d from %s" % (start, ckpts[-1]))
    if start >= tr["niter"]:
        print("already complete: iter %d >= niter %d" % (start, tr["niter"]))
        return

    # ---------------- data
    dc = cfg["data"]
    ds = D.TrainSet(dc.get("root", os.path.join(paths.EXPDATA, "train")), dc["hr_patch"], dc.get("scale", 4),
                    dc.get("roughness_mode", "lean"))
    loader = torch.utils.data.DataLoader(ds, batch_size=tr["batch"], shuffle=True,
                                         num_workers=dc.get("workers", 6), collate_fn=D.collate,
                                         drop_last=True, pin_memory=True, persistent_workers=True)
    it_data = infinite(loader)
    eval_set = sorted(os.path.basename(os.path.dirname(p)) for p in
                      glob.glob(os.path.join(paths.MATERIALS, "eval12", "*", "meta.json")))
    print("run %s | G %.2fM params (%s/%s) | D %s %s | %d train materials | eval %d | init %s"
          % (name, count_params(G) / 1e6, mc["arch"], mc["input"], dmode, gan_maps,
             len(ds.names), len(eval_set), init_report.get("loaded", "scratch")))
    json.dump({"init": init_report, "train_materials": len(ds.names)},
              open(os.path.join(run, "setup.json"), "w"), indent=2)

    log_f = open(os.path.join(run, "log.jsonl"), "a")
    eval_f = open(os.path.join(run, "eval.jsonl"), "a")
    torch.cuda.reset_peak_memory_stats(dev)
    clock = {"mark": time.perf_counter(), "seg_time": 0.0, "seg_iters": 0, "train": train_seconds}

    def flush_time():
        """Close the current training-time segment (eval/save time never enters)."""
        torch.cuda.synchronize()
        now = time.perf_counter()
        clock["seg_time"] += now - clock["mark"]
        clock["train"] += now - clock["mark"]
        clock["mark"] = now

    acc = {}
    for it in range(start + 1, tr["niter"] + 1):
        lr_b, hr_b = next(it_data)
        lr_b = {m: v.to(dev, non_blocking=True) for m, v in lr_b.items()}
        hr_b = {m: v.to(dev, non_blocking=True) for m, v in hr_b.items()}
        psnr_stage = it <= tr.get("psnr_iters", 0)

        for p in discs.parameters():
            p.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            sr = G(lr_b)
            loss_g, logs = crit.generator(sr, hr_b, discs, psnr_stage)
        loss_g.backward()
        opt_g.step()
        sch_g.step()

        if opt_d is not None and not psnr_stage:
            for p in discs.parameters():
                p.requires_grad_(True)
            opt_d.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                loss_d = discs.loss(sr, hr_b)
            loss_d.backward()
            opt_d.step()
            logs["D"] = loss_d
        if sch_d is not None:
            sch_d.step()

        # accumulate on the GPU: float() here would synchronise every step and
        # make it/s (and GPU-hours, experiment 5) a measure of the logging
        for k, v in logs.items():
            acc[k] = acc.get(k, 0.0) + v.detach().float()
        acc["G"] = acc.get("G", 0.0) + loss_g.detach().float()
        clock["seg_iters"] += 1
        last = it == tr["niter"]

        if it % cfg["log"]["every"] == 0 or last:
            flush_time()
            n = clock["seg_iters"]
            means = {k: float(v) / n for k, v in acc.items()}
            if not all(np.isfinite(v) for v in means.values()):
                raise SystemExit("non-finite loss in iters %d-%d: %s" % (it - n + 1, it, means))
            rec = {"iter": it, "stage": "psnr" if psnr_stage else "gan",
                   "it_per_s": n / max(clock["seg_time"], 1e-9), "train_seconds": clock["train"],
                   "lr_g": sch_g.get_last_lr()[0],
                   "peak_mem_mb": torch.cuda.max_memory_allocated(dev) / 2 ** 20, **means}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print("[%6d] %s G %.4f  %.2f it/s  %.0f MB" % (it, rec["stage"], rec["G"], rec["it_per_s"],
                                                           rec["peak_mem_mb"]), flush=True)
            acc, clock["seg_time"], clock["seg_iters"] = {}, 0.0, 0

        if it % cfg["eval"]["every"] == 0 or last:
            flush_time()
            ev = quick_eval(G, eval_set, dev, amp)
            eval_f.write(json.dumps({"iter": it, "train_seconds": clock["train"], **ev}) + "\n")
            eval_f.flush()
            print("  eval  " + "  ".join("%s %.4f" % kv for kv in ev.items()), flush=True)
            clock["mark"] = time.perf_counter()

        if it % cfg["save"]["every"] == 0 or last:
            flush_time()
            torch.save({"G": G.state_dict(), "opt_g": opt_g.state_dict(), "sch_g": sch_g.state_dict(),
                        "D": discs.state_dict() if opt_d is not None else None,
                        "opt_d": opt_d.state_dict() if opt_d is not None else None,
                        "sch_d": sch_d.state_dict() if sch_d is not None else None,
                        "iter": it, "train_seconds": clock["train"], "config": cfg},
                       os.path.join(run, "ckpt", "iter_%d.pt" % it))
            old = sorted(glob.glob(os.path.join(run, "ckpt", "iter_*.pt")),
                         key=lambda p: int(p.split("_")[-1].split(".")[0]))
            for p in old[:-cfg["save"].get("keep", 3)]:
                os.remove(p)
            clock["mark"] = time.perf_counter()
    print("training done: %s" % run)


if __name__ == "__main__":
    main()
