"""Arm 2-0, the baseline (configs/2-0_base_scratch.yaml), and the skeleton every
other arm plugs into.

    G          SharedSR: ONE 3->3 RRDBNet; the five maps go through it as one
               batch of 5B images (Alonso & Patow style)
    init       scratch (nothing to do; pretrained is arms/pretrained.py)
    D          SharedD: ONE 3-channel VGG discriminator, shared by every map
               that has a GAN weight, called once per map
    Criterion  per map: L1 (pixel group), VGG, RaGAN against the shared D

Criterion is a fixed sequence of loss terms, evaluated in exactly the old
experiments/exp2_esrgan/losses.py:Criterion.generator order (the float sum is
order dependent and has to stay bit-identical):

    for every map m in MAPS:
        pixel group     L1Term (this file), AngularTerm (2-3)    summed first, then added
        map terms       ExtraL1Term (3-1), VGGTerm, PerMapGanTerm (this file)
    post terms          JointGanTerm (3-2), RenderTerm (3-1)

Each term says whether it also runs in the PSNR warm-up (`in_psnr_stage`).
Pixel terms always run and switch their weight to 1.0 there. Of the others
only ExtraL1Term and RenderTerm run in the warm-up - the old trainer's
`if psnr_stage: continue` sat after extra_l1, and the render term after the
loop never looked at the stage. That is kept and named in
exp3_render_loss/arms/render_loss.py.

A term returns None (does not apply) or (weight, value); Criterion adds
weight * value to the total and logs value under the term's key.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from variants.experiments.exp2_esrgan.core.model import MAPS, RRDBNet


# ---------------------------------------------------------------- RaGAN

def ragan_g(d_real, d_fake):
    bce = F.binary_cross_entropy_with_logits
    real = d_real.detach()
    return 0.5 * (bce(real - d_fake.mean(), torch.zeros_like(real))
                  + bce(d_fake - real.mean(), torch.ones_like(d_fake)))


def ragan_d(d_real, d_fake):
    bce = F.binary_cross_entropy_with_logits
    return 0.5 * (bce(d_real - d_fake.mean().detach(), torch.ones_like(d_real))
                  + bce(d_fake - d_real.mean().detach(), torch.zeros_like(d_fake)))


# -------------------------------------------------------------------- G

class SharedSR(nn.Module):
    """input=independent, arch=shared: one 3->3 net, all maps through it.
    state_dict keys: net.*"""
    arch, input = "shared", "independent"

    def __init__(self, num_block=23, num_feat=64, num_grow_ch=32):
        super().__init__()
        mk = lambda i, o: RRDBNet(i, o, num_feat, num_block, num_grow_ch)
        self.net = mk(3, 3)

    def subnets(self):
        return [self.net]

    def forward(self, lr):
        y = self.net(torch.cat([lr[m] for m in MAPS], 0))
        return dict(zip(MAPS, torch.chunk(y, len(MAPS), dim=0)))


# -------------------------------------------------------------------- D

class Discriminators(torch.nn.Module):
    """The discriminator holder every arm uses. On its own it is the 'no D'
    case (disc.mode none, per_map without any GAN map, or an unknown mode -
    the old class silently built nothing for those too): no parameters,
    active() False, so the trainer creates no D optimiser.

    `mode`, `gan_maps` and `separate` are what the old class stored; Criterion
    terms read `mode` and call `for_map` at loss time."""

    def __init__(self, mode, gan_maps, separate):
        super().__init__()
        self.mode, self.gan_maps, self.separate = mode, list(gan_maps), separate

    def active(self):
        return (self.mode == "per_map" and bool(self.gan_maps)) or self.mode == "joint"

    def loss(self, sr, hr):
        sr = {m: v.detach() for m, v in sr.items()}
        return sum(ragan_d(self.for_map(m)(hr[m]), self.for_map(m)(sr[m])) for m in self.gan_maps)


class SharedD(Discriminators):
    """disc.mode=per_map with a shared generator: ONE D for every GAN map.
    state_dict keys: shared.*"""

    def __init__(self, mode, gan_maps, separate, make):
        super().__init__(mode, gan_maps, separate)
        self.shared = make(3)

    def for_map(self, m):
        return self.shared


# ------------------------------------------------------------ Criterion

class Criterion:
    """Generator objective: the ordered terms of the module docstring.
    Holds VGG and the discriminator(s)' G-side through its terms."""

    def __init__(self, per_map, pixel, map_terms, post_terms):
        self.per_map = per_map
        self.pixel, self.map_terms, self.post_terms = list(pixel), list(map_terms), list(post_terms)

    def pixel_terms(self, m, sr, hr, psnr_stage):
        spec, logs, total = self.per_map.get(m, {}), {}, 0.0
        for term in self.pixel:
            r = term(m, spec, sr, hr, psnr_stage)
            if r is not None:
                w, v = r
                total, logs[term.key] = total + w * v, v
        return total, logs

    def generator(self, sr, hr, discs, psnr_stage):
        total, logs = 0.0, {}
        for m in MAPS:
            t, lg = self.pixel_terms(m, sr[m], hr[m], psnr_stage)
            total = total + t
            logs.update({"%s/%s" % (m, k): v for k, v in lg.items()})
            for term in self.map_terms:
                if psnr_stage and not term.in_psnr_stage:
                    continue
                r = term(m, sr, hr, discs)
                if r is not None:
                    w, v = r
                    total, logs["%s/%s" % (m, term.key)] = total + w * v, v
        for term in self.post_terms:
            if psnr_stage and not term.in_psnr_stage:
                continue
            r = term(sr, hr, discs)
            if r is not None:
                w, v = r
                total, logs[term.key] = total + w * v, v
        return total, logs


class L1Term:
    """Pixel group. L1 on the encoded 3-channel image, for every map whose
    spec has an `l1` key - with weight 1.0 in the PSNR warm-up, whatever the
    config says (ESRGAN's PSNR-oriented stage)."""
    key = "l1"

    def __call__(self, m, spec, sr, hr, psnr_stage):
        if "l1" not in spec:
            return None
        v = F.l1_loss(sr.float(), hr)
        w = 1.0 if psnr_stage else spec["l1"]
        return w, v


class VGGTerm:
    """ESRGAN perceptual loss on maps with vgg > 0. GAN stage only."""
    key = "vgg"
    in_psnr_stage = False

    def __init__(self, per_map, vgg):
        self.per_map, self.vgg = per_map, vgg

    def __call__(self, m, sr, hr, discs):
        spec = self.per_map.get(m, {})
        if spec.get("vgg", 0) > 0:
            v = self.vgg(sr[m].float(), hr[m])
            return spec["vgg"], v
        return None


class PerMapGanTerm:
    """RaGAN, G side, against the map's discriminator (the shared one, or the
    map's own under 2-2) - only while the discriminators are per_map. Under
    disc.mode joint (3-2) these weights are inert: see joint_d.py."""
    key = "gan_g"
    in_psnr_stage = False

    def __init__(self, per_map):
        self.per_map = per_map

    def __call__(self, m, sr, hr, discs):
        spec = self.per_map.get(m, {})
        if spec.get("gan", 0) > 0 and discs.mode == "per_map":
            D = discs.for_map(m)
            v = ragan_g(D(hr[m]), D(sr[m]))
            return spec["gan"], v
        return None
