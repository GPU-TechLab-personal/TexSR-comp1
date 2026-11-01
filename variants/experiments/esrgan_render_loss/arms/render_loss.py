"""Arm 3-1 (configs/3-1_render.yaml): + rendering loss, + weak per-map L1.

    RenderTerm    k of MatUp's 100 Fibonacci lights drawn per iteration, SR and
                  GT maps rendered with MatUp's renderer on the same patch grid,
                  L1 in display space (metric l1) or log-L1 on linear radiance
                  (metric logl1). Normals are renormalised before rendering
                  (decode_t does it). Active when loss.render.weight > 0.
    ExtraL1Term   loss.extra_l1: {map: weight}, an L1 ON TOP of the per-map one
                  (3-1 gives basecolor/normal/roughness 0.01 - the same weight
                  2-0's l1 already has, so those maps get 0.02 in the GAN stage).

KEPT, NAMED: BOTH TERMS ALSO RUN IN THE PSNR WARM-UP (in_psnr_stage = True).
The old Criterion.generator applied extra_l1 before its `if psnr_stage:
continue` and the render term after the per-map loop, without looking at the
stage (experiments/exp2_esrgan/losses.py:118-121, 134-136). So in 3-1/3-2 the
first train.psnr_iters (20k) iterations optimise

    sum_m L1_m (weight 1.0)  +  sum_m extra_l1[m] * L1_m  +  render.weight * L_render

- not "pixel terms only" as ESRGAN's warm-up (and losses.py's docstring) has
it. The render term is at full weight from iteration 1. Changing that would
change every 3-x result; it is a decision for the experiment, not the refactor.
"""
import torch
import torch.nn.functional as F

from variants.experiments.common import render as R
from variants.experiments.esrgan_map_degradation.core.dataset import decode_t


def srgb_to_linear_t(x):
    return torch.where(x > 0.04045, ((x.clamp_min(0.04045) + 0.055) / 1.055) ** 2.4, x / 12.92)


def render_tensors(canon, grid):
    """decoded canonical tensors (B,H,W[,3]) -> MatUp renderer inputs, patch grid [-1,1]."""
    n = canon["normal"].clone()
    n[..., 1] = -n[..., 1]                                            # OpenGL -> MatUp internal
    alpha = canon["roughness"].clamp_min(R.MIN_ROUGHNESS) ** 2
    B = n.shape[0]
    pos = torch.cat([grid.expand(B, -1, -1, -1),
                     (R.DISP_VAL * (canon["height"] - 0.5))[..., None]], dim=-1)
    return [srgb_to_linear_t(canon["basecolor"]), canon["metallic"][..., None], n,
            alpha[..., None], pos]


class RenderLoss:
    def __init__(self, cfg, device, patch):
        self.w = float(cfg.get("weight", 0))
        self.k = int(cfg.get("k", 8))
        self.metric = cfg.get("metric", "l1")
        if self.metric not in ("l1", "logl1"):
            raise ValueError(self.metric)
        R.init(device)
        self.lights = R.fib_lights(int(cfg.get("n_lights", 100)), device)
        self.grid = R.position_grid(patch, 0, device)[None]
        self.device = device

    def __call__(self, sr_enc, hr_enc):
        idx = torch.randperm(self.lights.shape[0], device=self.device)[:self.k]
        wi = self.lights[idx]
        tonemap = self.metric == "l1"
        with torch.autocast("cuda", enabled=False):
            a = R.render(render_tensors(decode_t(sr_enc), self.grid), wi, tonemap, chunk=self.k)
            with torch.no_grad():
                b = R.render(render_tensors(decode_t(hr_enc), self.grid), wi, tonemap, chunk=self.k)
            if self.metric == "l1":
                return F.l1_loss(a, b)
            return (torch.log(a + 0.01) - torch.log(b + 0.01)).abs().mean()


def applies(loss_cfg):
    """loss.render.weight > 0 - the condition the old trainer built RenderLoss under."""
    return loss_cfg.get("render", {}).get("weight", 0) > 0


class ExtraL1Term:
    """Map term: extra_l1[m] * L1, for every map listed in loss.extra_l1."""
    key = "extra_l1"
    in_psnr_stage = True             # KEPT: runs in the PSNR warm-up too (module docstring)

    def __init__(self, extra_l1):
        self.extra_l1 = extra_l1

    def __call__(self, m, sr, hr, discs):
        if m in self.extra_l1:
            v = F.l1_loss(sr[m].float(), hr[m])
            return self.extra_l1[m], v
        return None


class RenderTerm:
    """Post term: render.weight * RenderLoss(sr, hr)."""
    key = "render"
    in_psnr_stage = True             # KEPT: runs in the PSNR warm-up too, at full weight

    def __init__(self, render):
        self.render = render

    def __call__(self, sr, hr, discs):
        if self.render.w > 0:
            v = self.render(sr, hr)
            return self.render.w, v
        return None
