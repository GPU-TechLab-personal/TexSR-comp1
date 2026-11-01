"""Arm 3-2 (configs/3-2_render_jointD.yaml, disc.mode: joint, loss.joint_gan):
one discriminator over the 15-channel concatenation of all maps (it sees
cross-map consistency) instead of one D per map.

KEPT, NAMED: under disc.mode joint the per-map GAN weights inherited from 2-0
(`gan: 0.005` on every map) do nothing - base.PerMapGanTerm only fires while
discs.mode == "per_map" (old losses.py:127). They still decide `gan_maps`, which
the trainer prints ("D joint [basecolor, ...]") but JointD ignores. The only
adversarial term of 3-2 is joint_gan * RaGAN(joint D).
"""
import torch

from variants.experiments.esrgan_map_degradation.arms.base import Discriminators, ragan_d, ragan_g
from variants.experiments.esrgan_map_degradation.core.model import MAPS


def cat(maps):
    return torch.cat([maps[m] for m in MAPS], 1)


class JointD(Discriminators):
    """disc.mode=joint: ONE 15-channel D. state_dict keys: joint.*"""

    def __init__(self, mode, gan_maps, separate, make):
        super().__init__(mode, gan_maps, separate)
        self.joint = make(15)

    def loss(self, sr, hr):
        sr = {m: v.detach() for m, v in sr.items()}
        return ragan_d(self.joint(cat(hr)), self.joint(cat(sr)))


class JointGanTerm:
    """Post term: joint_gan * RaGAN_G against the joint D. GAN stage only."""
    key = "joint/gan_g"
    in_psnr_stage = False

    def __init__(self, joint_gan):
        self.joint_gan = joint_gan

    def __call__(self, sr, hr, discs):
        if discs.mode == "joint" and self.joint_gan > 0:
            v = ragan_g(discs.joint(cat(hr)), discs.joint(cat(sr)))
            return self.joint_gan, v
        return None
