#!/usr/bin/env python3
"""One ESRGAN / SwinIR arm -> the arm folder every evaluator reads (--arm TAG=DIR).

Runs in the srft service:

    # zero-shot: the official weight (nets.WEIGHTS), manifest step 0
    docker compose run --rm srft python variants/sr_finetune/write_arm.py \\
        --arm swinir_real_zs --out /workspace/runs/sr_ft/eval/swinir_real_zs
    # fine-tuned: one checkpoint of its trainer (nets.ARMS[...]["ft_file"])
    docker compose run --rm srft python variants/sr_finetune/write_arm.py \\
        --arm esrgan_ft --ckpt /workspace/runs/sr_ft/basicsr/<name>/models/net_g_40000.pth \\
        --step 40000 --out /workspace/runs/sr_ft/eval/esrgan_ft_40k

Everything but the model is variants/util/arm_output.py, the same path the
DeepTextures arms take (deeptex/matsynth/eval_arms.py):

  input    arm_output.load_as_rgb01 -> map_encode.to_rgb8: 8-bit RGB, a 1-channel
           map replicated to three channels, 16-bit (height, render_lr) reduced
           by rint(v/257)
  model    nets.build(arch) - the OFFICIAL network file, strict load
  predict  NHWC float32 [0,1] -> NCHW torch -> model -> NHWC, float32 on the GPU,
           no clamp (save_sr clips; write_arm sees the raw output for its
           non-finite check)
  output   arm_output.save_sr(..., "mean"): colour maps as 3-channel 8-bit, the
           1-channel maps as the mean of the three output channels; the
           channel spread goes into sr_manifest.json

SWINIR PADDING. main_test_swinir.py:70-73 mirror-pads the input to a multiple of
the window (8) and crops the output back - and its formula,
(h // 8 + 1) * 8 - h, adds a FULL window when h is already a multiple: a 128^2
map is run at 136^2. predict() does exactly that, so the arm is what the official
script would produce (gates/gate_parity.py holds it to 1 LSB).

Exit codes: 2 bad arguments / no materials, 4 a non-finite output (write_arm),
10 CUDA out of memory (lower --tile-size).
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 2)))
from variants.util import arm_output, paths  # noqa: E402
from variants.sr_finetune import nets  # noqa: E402

USE = {"basicsr": paths.use_basicsr, "swinir": paths.use_swinir_official}
SCALE = 4


def die(code, msg):
    print("[fatal] %s" % msg)
    sys.exit(code)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(tag, ckpt, step):
    """(arch, weight file, state_dict key, step) for an arm tag."""
    if tag not in nets.ARMS:
        die(2, "unknown --arm %s (nets.ARMS: %s)" % (tag, ", ".join(nets.ARMS)))
    arm = nets.ARMS[tag]
    w = nets.WEIGHTS[arm["weights"]]
    if "ft_file" not in arm:
        if ckpt:
            die(2, "%s is zero-shot; it takes no --ckpt" % tag)
        return w["arch"], w["file"], w["key"], 0
    if not ckpt or step is None:
        die(2, "%s needs --ckpt and --step (its trainer writes %s)" % (tag, arm["ft_file"]))
    if os.path.basename(ckpt) != arm["ft_file"].format(step=step):
        die(2, "--ckpt %s is not step %d's %s" % (ckpt, step, arm["ft_file"].format(step=step)))
    return w["arch"], ckpt, arm["ft_key"], step


def predictor(net, window, device):
    import numpy as np
    import torch

    def predict(batch):
        x = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).to(device)
        h, w = x.shape[2], x.shape[3]
        if window:
            # main_test_swinir.py:70-73, verbatim arithmetic (a full window even on a multiple)
            h_pad = (h // window + 1) * window - h
            w_pad = (w // window + 1) * window - w
            x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h + h_pad, :]
            x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w + w_pad]
        with torch.no_grad():
            y = net(x)
        y = y[..., :h * SCALE, :w * SCALE]
        return y.permute(0, 2, 3, 1).float().cpu().numpy()
    return predict


def load_model(arch, path, key):
    """The official network with `path` loaded strictly, on the GPU, in eval mode."""
    import torch
    USE[nets.ARCHS[arch]["code"]]()
    net = nets.build(arch)
    net.load_state_dict(nets.load_state(path, key), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return net.eval().to(device), device


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="a tag in nets.ARMS")
    ap.add_argument("--ckpt", default="", help="fine-tuned arms: the checkpoint file")
    ap.add_argument("--step", type=int, default=None, help="fine-tuned arms: its step")
    ap.add_argument("--root", default="/workspace/runs/matsynth")
    ap.add_argument("--out", required=True, help="e.g. /workspace/runs/sr_ft/eval/<tag>[_NNk]")
    ap.add_argument("--material", default="")
    ap.add_argument("--maps", default="all",
                    help="comma-separated subset of %s, or 'all'" % ",".join(j[0] for j in arm_output.ALL_JOBS))
    ap.add_argument("--run", default="", help="MatUp output stamp; default is the newest")
    ap.add_argument("--tile-size", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    arch, path, key, step = resolve(args.arm, args.ckpt, args.step)
    if not os.path.exists(path):
        die(2, "no weight file %s" % path)
    jobs = arm_output.select(arm_output.ALL_JOBS, args.maps)
    if not jobs:
        die(2, "--maps matched nothing")
    mats = arm_output.materials(args.root, args.material)
    if not mats:
        die(2, "no materials under %s" % args.root)

    import torch
    net, device = load_model(arch, path, key)
    predict = predictor(net, nets.ARCHS[arch]["kwargs"].get("window_size"), device)

    def sr_fn(lr, where):
        try:
            return arm_output.super_resolve(predict, lr, SCALE, args.tile_size, args.overlap)
        except torch.cuda.OutOfMemoryError as e:
            die(10, "OOM on %s - lower --tile-size\n%s" % (where, e))

    provenance = {"checkpoint": os.path.splitext(os.path.basename(path))[0], "step": step,
                  "sub_name": args.arm, "scale": SCALE,
                  "checkpoint_file": os.path.abspath(path), "weight_sha": sha256(path),
                  "param_key": key, "arch": arch,
                  "network_file": os.path.relpath(sys.modules[nets.ARCHS[arch]["module"]].__file__, paths.REPO)}
    done, skipped = arm_output.write_arm(
        mats, args.out, jobs, sr_fn,
        lambda p, sr, kind: arm_output.save_sr(p, sr, kind, "mean"),
        ("mean", arm_output.COLLAPSE["mean"]), provenance, run=args.run, die=die)
    print("%d material(s), %d map(s) each -> %s  (%s, step %d)"
          % (len(done), len(jobs), args.out, provenance["checkpoint"], step))
    for s in skipped[:20]:
        print("  [skip] %s" % s)
    if len(skipped) > 20:
        print("  [skip] ... %d more" % (len(skipped) - 20))


if __name__ == "__main__":
    main()
