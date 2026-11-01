#!/usr/bin/env python3
"""The official starting weights: check and record them, and hand them to KAIR.

Runs in the srft service (torch is needed to read the keys):

    docker compose run --rm srft python variants/sr_finetune/weights.py --check
    docker compose run --rm srft python variants/sr_finetune/weights.py \\
        --kair-init swinir_real --models-dir runs/sr_ft/kair/<task>/models

--check      every file in nets.WEIGHTS: present, sha256 (and the publisher's
             prefix where the name carries one), the state_dict key, tensor and
             parameter counts -> runs/sr_ft/weights/manifest.json. Exit 2 if any
             is missing or its sha does not match.

--kair-init  KAIR does not read "pretrained_netG" from its json at the start:
             main_train_psnr.py:63 and main_train_gan.py:61 call
             find_last_checkpoint() without pretrained_path, so an empty models/
             folder gives pretrained_netG = None and training starts from random
             weights (utils/utils_option.py:191-193). It starts from the
             highest-numbered <N>_G.pth / <N>_E.pth in models/ instead, so the
             official weights go in as 0_G.pth and 0_E.pth (step 0, so the
             learning-rate schedule starts at the beginning).
             They are written as a bare state_dict: model_base.load_network looks
             for 'params' only (models/model_base.py:165), and the real-world file
             has only 'params_ema'. G and E both get the released weights - for
             swinir_real that is the EMA, the only generator released. No 0_D.pth:
             the discriminator starts from scratch, as in the official recipe.
             Refuses a models/ folder that already holds a *_G.pth or *_E.pth, so
             a resume is never overwritten.

BasicSR needs no such step: its yml names the file (pretrain_network_g), and
load_network falls back from 'params_ema' to 'params' (basicsr/models/base_model.py:292-294).
"""
import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 2)))
from variants.util import paths  # noqa: E402
from variants.sr_finetune import nets  # noqa: E402

MANIFEST = os.path.join(nets.WEIGHTS_DIR, "manifest.json")
KAIR_INIT = ("swinir_cls", "swinir_real")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check():
    rows, bad = {}, []
    for name, w in nets.WEIGHTS.items():
        row = {"file": os.path.relpath(w["file"], paths.REPO), "source": w["source"],
               "key": w["key"], "arch": w["arch"]}
        if not os.path.exists(w["file"]):
            row["missing"] = True
            bad.append("%s: missing %s" % (name, row["file"]))
            rows[name] = row
            continue
        row["bytes"] = os.path.getsize(w["file"])
        row["sha256"] = sha256(w["file"])
        if w["sha_prefix"] and not row["sha256"].startswith(w["sha_prefix"]):
            bad.append("%s: sha256 %s does not start with %s" % (name, row["sha256"][:12], w["sha_prefix"]))
        state = nets.load_state(w["file"], w["key"])
        row["tensors"] = len(state)
        row["parameters"] = int(sum(v.numel() for v in state.values()))
        rows[name] = row
        print("  %-13s %10d B  sha256 %s  %s %d tensors" % (
            name, row["bytes"], row["sha256"][:12], w["key"] or "(bare)", row["tensors"]))
    os.makedirs(nets.WEIGHTS_DIR, exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump({"weights": rows, "problems": bad}, f, indent=2)
    for b in bad:
        print("[fatal] " + b)
    print("manifest -> %s" % MANIFEST)
    sys.exit(2 if bad else 0)


def kair_init(arm, models_dir):
    import torch
    taken = glob.glob(os.path.join(models_dir, "*_G.pth")) + glob.glob(os.path.join(models_dir, "*_E.pth"))
    if taken:
        raise SystemExit("[fatal] %s already holds %s; refusing to overwrite a run"
                         % (models_dir, ", ".join(sorted(os.path.basename(t) for t in taken))))
    w = nets.WEIGHTS[arm]
    state = nets.load_state(w["file"], w["key"])
    os.makedirs(models_dir, exist_ok=True)
    for net in ("G", "E"):
        torch.save(state, os.path.join(models_dir, "0_%s.pth" % net))
    with open(os.path.join(models_dir, "0_init.json"), "w") as f:
        json.dump({"arm": arm, "from": os.path.relpath(w["file"], paths.REPO),
                   "sha256": sha256(w["file"]), "key": w["key"], "written": ["0_G.pth", "0_E.pth"]},
                  f, indent=2)
    print("%s: %s['%s'] -> %s/0_G.pth, 0_E.pth" % (arm, os.path.basename(w["file"]), w["key"], models_dir))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--kair-init", choices=KAIR_INIT)
    ap.add_argument("--models-dir")
    args = ap.parse_args()
    if args.kair_init:
        if not args.models_dir:
            ap.error("--kair-init needs --models-dir")
        return kair_init(args.kair_init, args.models_dir)
    if args.check:
        return check()
    ap.error("--check or --kair-init")


if __name__ == "__main__":
    main()
