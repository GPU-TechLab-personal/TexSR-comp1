#!/usr/bin/env python3
"""Colour drift of a fine-tuned arm, checkpoint by checkpoint - the torch driver of
variants/util/report/drift.py (deeptex/matsynth/deeptex_drift.py is the TF one).

Runs in the srft service:

    docker compose run --rm srft python variants/sr_finetune/drift.py \\
        --arm esrgan_ft --ckpt-dir /workspace/runs/sr_ft/basicsr/experiments/esrgan_ft/models \\
        --pairs /workspace/runs/deeptex/datasets/matsynth/baseColor --eval-map baseColor \\
        --sheet-material 0844_syntetic_foam --out /workspace/runs/sr_ft/drift/esrgan_ft_baseColor

Every checkpoint of the arm in --ckpt-dir (nets.ARMS[arm]["ft_file"], in step order;
the zero-shot weight is added as step 0 so the table starts where training did)
goes through write_arm.py's own model and predict(), then drift.offsets /
summarise / sheet_outputs / write_sheet / write_report: the same definition as the
DeepTextures drift tables (train = --n seeded pairs of --pairs, the model's own
training data; eval = the evaluation materials' --eval-map), so the two read side
by side.
"""
import argparse
import glob
import os
import random
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 2)))
from variants.sr_finetune import nets, write_arm  # noqa: E402
from variants.util.report import drift  # noqa: E402


def checkpoints(arm, ckpt_dir):
    """[(step, path)] in step order, the zero-shot weight first as step 0."""
    pattern = re.escape(nets.ARMS[arm]["ft_file"]).replace(r"\{step\}", r"(\d+)") + "$"
    found = []
    for p in glob.glob(os.path.join(ckpt_dir, "*.pth")):
        m = re.match(pattern, os.path.basename(p))
        if m and int(m.group(1)) > 0:
            found.append((int(m.group(1)), p))
    if not found:
        raise SystemExit("[fatal] no %s under %s" % (nets.ARMS[arm]["ft_file"], ckpt_dir))
    return [(0, nets.WEIGHTS[nets.ARMS[arm]["weights"]]["file"])] + sorted(found)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=[t for t, a in nets.ARMS.items() if "ft_file" in a])
    ap.add_argument("--ckpt-dir", required=True, help="the arm's checkpoint_root (arms.json)")
    ap.add_argument("--pairs", required=True, help="a folder holding HR/ and LRx4/")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-root", default="/workspace/runs/matsynth")
    ap.add_argument("--eval-map", default="baseColor", help="file stem under SVBRDF/ and GT/")
    ap.add_argument("--sheet-material", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arm = nets.ARMS[args.arm]
    zs = nets.WEIGHTS[arm["weights"]]
    names = sorted(f for f in os.listdir(os.path.join(args.pairs, "HR")) if f.endswith(".png"))
    pick = random.Random(args.seed).sample(names, min(args.n, len(names)))
    train = [(os.path.join(args.pairs, "LRx4", n), os.path.join(args.pairs, "HR", n)) for n in pick]
    mats = sorted(d for d in glob.glob(os.path.join(args.eval_root, "*"))
                  if os.path.isdir(os.path.join(d, "SVBRDF")))
    ev = [(os.path.join(d, "SVBRDF", args.eval_map + ".png"), os.path.join(d, "GT", args.eval_map + ".png"))
          for d in mats]
    os.makedirs(args.out, exist_ok=True)

    net, device = write_arm.load_model(zs["arch"], zs["file"], zs["key"])
    predict = write_arm.predictor(net, nets.ARCHS[zs["arch"]]["kwargs"].get("window_size"), device)
    rows = []
    for step, path in checkpoints(args.arm, args.ckpt_dir):
        key = zs["key"] if step == 0 else arm["ft_key"]
        net.load_state_dict(nets.load_state(path, key), strict=True)
        ck = os.path.splitext(os.path.basename(path))[0] if step else "zero-shot"
        row = {"checkpoint": ck, "step": step,
               "train": drift.summarise(drift.offsets(predict, train)),
               "eval": drift.summarise(drift.offsets(predict, ev))}
        if args.sheet_material:
            mat_dir = os.path.join(args.eval_root, args.sheet_material)
            outs = drift.sheet_outputs(predict, mat_dir, os.path.join(args.out, "outputs", ck, args.sheet_material))
            when = "%dk" % (step // 1000) if step and step % 1000 == 0 else str(step)
            drift.write_sheet(os.path.join(args.out, "%s_%s_%s.png" % (args.sheet_material, ck, when)),
                              "%s    %s  step %d" % (args.sheet_material, ck, step), [ck, when], outs)
        rows.append(row)
        drift.print_row(row)

    drift.write_report(args.out, {"arm": args.arm, "ckpt_dir": args.ckpt_dir, "pairs": args.pairs,
                                  "n": len(train), "seed": args.seed, "eval_root": args.eval_root,
                                  "eval_map": args.eval_map},
                       rows, len(train), len(ev), args.pairs, args.seed, args.eval_map, args.ckpt_dir)


if __name__ == "__main__":
    main()
