"""Argument plumbing shared by the experiment entry points."""
import glob
import json
import os

import torch

from . import paths, upscalers


def add_material_args(ap, default_set="exp1"):
    ap.add_argument("--set", default=default_set, help="material set under /expdata/materials")
    ap.add_argument("--materials", nargs="*", help="explicit names (default: every material in the set)")
    ap.add_argument("--pilot", action="store_true", help="only the selection's pilot materials")
    ap.add_argument("--selection", default="/workspace/experiments/data/selection_exp1.json")
    ap.add_argument("--limit", type=int, default=0)


def resolve_materials(args):
    root = os.path.join(paths.MATERIALS, args.set)
    have = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(root, "*", "meta.json")))
    if args.materials:
        names = args.materials
    elif args.pilot:
        names = json.load(open(args.selection))["pilot"]
    else:
        names = have
    missing = [n for n in names if n not in have]
    if missing:
        raise SystemExit("not extracted under %s: %s (run experiments/data/extract.py)" % (root, missing))
    return names[:args.limit] if args.limit else names


def add_s_args(ap, default=("swinir",)):
    ap.add_argument("--s", nargs="+", default=list(default),
                    help="upsamplers: %s" % ", ".join(sorted(upscalers.REGISTRY)))
    ap.add_argument("--s-opt", nargs="*", default=[],
                    help="per-upsampler settings, e.g. sdx4.steps=20 sdx4.seed_mode=shared srflow.heat=0.5")


def iter_upscalers(args, device):
    """Build one S at a time (SwinIR-L and the diffusion pipeline together
    would not fit next to MatUp on a 12 GB card) and free it afterwards."""
    for name in args.s:
        opts = upscalers.parse_opts([o.split(".", 1)[1] for o in args.s_opt
                                     if o.startswith(name + ".")])
        try:
            up = upscalers.get(name, device, **opts)
        except upscalers.UpscalerUnavailable as exc:
            print("SKIP S=%s: %s" % (name, exc))
            continue
        yield up
        del up
        if device.type == "cuda":
            torch.cuda.empty_cache()


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
