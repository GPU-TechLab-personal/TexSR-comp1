"""Arm 2-2 (configs/2-2_separate.yaml, model.arch: separate): one network per map.

The config changes ONE key, but the arm changes TWO things - kept as it was,
and spelled out here:

  1. FIVE generators, one 3->3 RRDBNet per map (SeparateSR below).
  2. FIVE discriminators, one per GAN map (PerMapD below), instead of the one
     shared D every other per_map arm has. Nothing in the config says so: the
     old trainer passed `mc["arch"] == "separate"` into Discriminators
     (experiments/exp2_esrgan/train.py:144) and the D class branched on it
     (losses.py:152-155). owns_discriminators() is that coupling, by name.

So 2-2 vs 2-0 is "separate G AND separate D", not "separate G" alone, and its
cost (1.41 it/s vs 4.1-6.4, experiments/README.md 8) includes both.
"""
import torch
import torch.nn as nn

from variants.experiments.esrgan_map_degradation.arms.base import Discriminators
from variants.experiments.esrgan_map_degradation.core.model import MAPS, RRDBNet


class SeparateSR(nn.Module):
    """input=independent, arch=separate: one 3->3 net per map.
    state_dict keys: nets.<map>.*"""
    arch, input = "separate", "independent"

    def __init__(self, num_block=23, num_feat=64, num_grow_ch=32):
        super().__init__()
        mk = lambda i, o: RRDBNet(i, o, num_feat, num_block, num_grow_ch)
        # five generators, built in MAPS order (the order the old dict
        # comprehension drew their initial weights in)
        self.nets = nn.ModuleDict({
            "basecolor": mk(3, 3),
            "normal": mk(3, 3),
            "roughness": mk(3, 3),
            "metallic": mk(3, 3),
            "height": mk(3, 3),
        })
        assert list(self.nets) == MAPS

    def subnets(self):
        return list(self.nets.values())

    def forward(self, lr):
        return {m: self.nets[m](lr[m]) for m in MAPS}


def owns_discriminators(model_cfg):
    """True when the generator is per map - and then so are the discriminators.
    The hidden second factor of 2-2 (module docstring)."""
    return model_cfg["arch"] == "separate"


class PerMapD(Discriminators):
    """disc.mode=per_map under arch=separate: one D per GAN map - five for
    2-2, which keeps gan > 0 on all five maps.
    state_dict keys: ds.<map>.*"""

    def __init__(self, mode, gan_maps, separate, make):
        super().__init__(mode, gan_maps, separate)
        self.ds = torch.nn.ModuleDict({m: make(3) for m in self.gan_maps})

    def for_map(self, m):
        return self.ds[m]
