"""Arm 2-1 (configs/2-1_base_pretrained.yaml, init.mode: pretrained): the
generator starts from a natural-image RRDBNet instead of scratch. Also used by
3-0/3-1/3-2, whose 15-channel head cannot take the 3-channel conv_first /
conv_last - those stay freshly initialised and are listed in setup.json
("skipped"), as before.

Default weights: RealESRGAN_x4plus (GitHub release). The official ESRGAN .pth
(Google Drive only) loads too, its old parameter names remapped by _OLD.
"""
import os
import re

import torch

PRETRAINED_URLS = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
}

_OLD = [(r"^RRDB_trunk\.", "body."), (r"\.RDB(\d)\.", r".rdb\1."), (r"^trunk_conv\.", "conv_body."),
        (r"^upconv1\.", "conv_up1."), (r"^upconv2\.", "conv_up2."), (r"^HRconv\.", "conv_hr.")]


def _read_state(path):
    sd = torch.load(path, map_location="cpu")
    for k in ("params_ema", "params", "state_dict"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
            break
    out = {}
    for k, v in sd.items():
        for pat, rep in _OLD:
            k = re.sub(pat, rep, k)
        out[k] = v
    return out


def ensure_pretrained(name_or_path, cache_dir):
    if os.path.isfile(name_or_path):
        return name_or_path
    if name_or_path not in PRETRAINED_URLS:
        raise FileNotFoundError("%s is neither a file nor one of %s" % (name_or_path, list(PRETRAINED_URLS)))
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, name_or_path + ".pth")
    if not os.path.isfile(dst):
        torch.hub.download_url_to_file(PRETRAINED_URLS[name_or_path], dst)
    return dst


def load_pretrained(model, path):
    """Copy every tensor whose name and shape match into each sub-network.
    With 15-channel concat input, conv_first / conv_last do not match and stay
    freshly initialised - that is 실험 3's 'head 새로 초기화', reported, not hidden."""
    sd = _read_state(path)
    report = {"path": path, "loaded": 0, "skipped": [], "missing": []}
    for net in model.subnets():
        own = net.state_dict()
        take = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        report["skipped"] = sorted(set(report["skipped"]) | {k for k in sd if k in own and k not in take})
        report["missing"] = sorted(set(report["missing"]) | {k for k in own if k not in sd})
        own.update(take)
        net.load_state_dict(own)
        report["loaded"] += len(take)
    if report["missing"]:
        raise RuntimeError("checkpoint lacks %d RRDBNet tensors (wrong architecture?): %s"
                           % (len(report["missing"]), report["missing"][:5]))
    return report


def load(G, weights, cache_dir):
    """The init step of experiments/exp2_esrgan/train.py:135-137."""
    path = ensure_pretrained(weights, cache_dir)
    return load_pretrained(G, path)
