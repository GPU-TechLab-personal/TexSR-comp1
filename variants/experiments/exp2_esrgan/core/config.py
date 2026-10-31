"""Config loading for experiments 2 and 3 (moved from experiments/exp2_esrgan/train.py:49-80).

A config names its parent with `base:` (a path relative to the config's own
folder); `--set a.b=v` overrides are yaml-parsed. The keys are the old ones -
the arm a config gets is read off them in core/assemble.py, never named in the
file.
"""
import copy

import os
import yaml


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
