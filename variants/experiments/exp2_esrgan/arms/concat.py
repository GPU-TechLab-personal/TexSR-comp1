"""Arm 2-4 (configs/2-4_concat_input.yaml, model.input: concat): all maps as
one 15-channel image through one 15->15 RRDBNet (inter-map information).
Also the generator of 3-0/3-1/3-2.

Only the generator changes. The discriminators stay per map (one shared 3-ch
D, base.SharedD) unless disc.mode says joint (3-2).
"""
import torch
import torch.nn as nn

from variants.experiments.exp2_esrgan.core.model import MAPS, RRDBNet


class ConcatSR(nn.Module):
    """input=concat: one 15->15 net over all maps at once.
    state_dict keys: net.*"""
    arch, input = "shared", "concat"

    def __init__(self, num_block=23, num_feat=64, num_grow_ch=32):
        super().__init__()
        mk = lambda i, o: RRDBNet(i, o, num_feat, num_block, num_grow_ch)
        self.net = mk(15, 15)

    def subnets(self):
        return [self.net]

    def forward(self, lr):
        y = self.net(torch.cat([lr[m] for m in MAPS], 1))
        return dict(zip(MAPS, torch.split(y, 3, dim=1)))
