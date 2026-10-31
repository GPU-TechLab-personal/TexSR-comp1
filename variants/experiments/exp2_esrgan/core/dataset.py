"""Training data and the map <-> 3-channel image encoding (see core/model.py).

Moved from experiments/exp2_esrgan/data.py unchanged (only the imports).
"""
import glob
import json
import os

import numpy as np
import torch

from variants.experiments.common.protocol import downsample, flip_rot, load_material
from variants.experiments.exp2_esrgan.core.model import MAPS


# ------------------------------------------------------------ encode/decode

def encode_np(mat):
    """canonical numpy material -> {map: (3, H, W) float32 in [0,1]}"""
    out = {}
    for m in MAPS:
        a = mat[m]
        if m == "normal":
            a = 0.5 * a + 0.5
        elif a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=-1)
        out[m] = np.ascontiguousarray(np.clip(a, 0, 1).transpose(2, 0, 1), dtype=np.float32)
    return out


def decode_t(maps):
    """{map: (B, 3, H, W)} network output -> canonical tensors (B, H, W[, 3])."""
    out = {}
    for m in MAPS:
        x = maps[m].float()
        if m == "normal":
            n = x * 2 - 1
            out[m] = (n / n.norm(dim=1, keepdim=True).clamp_min(1e-6)).permute(0, 2, 3, 1)
        elif m == "basecolor":
            out[m] = x.clamp(0, 1).permute(0, 2, 3, 1)
        else:
            out[m] = x.mean(dim=1).clamp(0, 1)
    return out


def decode_np(maps):
    """{map: (3, H, W) tensor} for ONE material -> canonical numpy."""
    d = decode_t({m: v[None] for m, v in maps.items()})
    return {m: v[0].detach().cpu().numpy() for m, v in d.items()}


def to_batch(mat_list, device):
    """[encoded dict] -> {map: (B,3,H,W)} on device"""
    return {m: torch.stack([torch.from_numpy(e[m]) for e in mat_list]).to(device) for m in MAPS}


# ------------------------------------------------------------------ dataset

class TrainSet(torch.utils.data.Dataset):
    """Random HR patches from /expdata/train/*/HR; LR by the unified protocol.

    Patch origin is a multiple of `scale`, so LR blocks line up with the
    block grid the evaluation LR uses. Augmentation: rot90 x flip, with the
    normal vectors rotated along (protocol.flip_rot)."""

    def __init__(self, root, hr_patch=128, scale=4, roughness_mode="lean",
                 patches_per_material=16, names=None, seed=0):
        listed = os.path.join(root, "materials.json")
        if names is None:
            # materials.json is written only when build_train_set.py COMPLETES.
            # Globbing the folders instead would silently train on whatever a
            # stopped build left behind.
            if not os.path.isfile(listed):
                built = len(glob.glob(os.path.join(root, "*", "meta.json")))
                raise RuntimeError("%s has no materials.json: the training set is incomplete (%d materials "
                                   "built). Run experiments/data/build_train_set.py to completion - it "
                                   "resumes where it stopped." % (root, built))
            names = json.load(open(listed))["materials"]
        if not names:
            raise RuntimeError("no training materials under %s (experiments/data/build_train_set.py)" % root)
        self.root, self.names = root, names
        self.hr_patch, self.scale, self.rmode = hr_patch, scale, roughness_mode
        self.ppm, self.seed = patches_per_material, seed

    def __len__(self):
        return len(self.names) * self.ppm

    def __getitem__(self, i):
        # KEPT: unseeded - patch positions and augmentation do not follow
        # train.seed, so two runs of one config never see the same batches
        # (variants/experiments/README.md, kept behaviour 6). `self.seed` is unused.
        rng = np.random.default_rng()
        mat = load_material(os.path.join(self.root, self.names[i % len(self.names)], "HR"))
        size = mat["basecolor"].shape[0]
        steps = (size - self.hr_patch) // self.scale + 1
        y, x = rng.integers(0, steps, 2) * self.scale
        crop = {k: v[y:y + self.hr_patch, x:x + self.hr_patch] for k, v in mat.items()}
        crop = flip_rot(crop, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))
        lr = downsample(crop, self.scale, self.rmode)
        return encode_np(lr), encode_np(crop)


def collate(batch):
    lr = {m: torch.stack([torch.from_numpy(b[0][m]) for b in batch]) for m in MAPS}
    hr = {m: torch.stack([torch.from_numpy(b[1][m]) for b in batch]) for m in MAPS}
    return lr, hr
