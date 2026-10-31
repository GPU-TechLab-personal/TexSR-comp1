"""
Losses for experiments 2 and 3, all configured per map.

Per-map spec (config `loss.per_map.<map>`), any subset of
    l1       L1 on the encoded 3-channel image
    vgg      ESRGAN perceptual (VGG19 conv5_4 pre-activation, L1)
    gan      relativistic-average GAN (ESRGAN), per-map discriminator
    angular  mean angle between decoded unit normals, radians (normal only)
Two stages, as ESRGAN: for it < psnr_iters only the pixel terms run
(l1 / angular, weight 1); afterwards the configured weights.

Rendering loss (실험 3): k of MatUp's 100 Fibonacci lights drawn per
iteration, SR and GT maps rendered with MatUp's renderer on the same patch
grid, L1 (display space) or log-L1 (linear radiance). Normals are
renormalised before rendering (decode_t does it).
"""
import torch
import torch.nn.functional as F

from experiments.common import render as R

from .data import decode_t
from .model import MAPS


def angular(pred_enc, gt_enc):
    p = F.normalize(pred_enc.float() * 2 - 1, dim=1)
    g = F.normalize(gt_enc.float() * 2 - 1, dim=1)
    cross = torch.linalg.norm(torch.cross(p, g, dim=1), dim=1)
    dot = (p * g).sum(1)
    return torch.atan2(cross, dot).mean()


def ragan_g(d_real, d_fake):
    bce = F.binary_cross_entropy_with_logits
    real = d_real.detach()
    return 0.5 * (bce(real - d_fake.mean(), torch.zeros_like(real))
                  + bce(d_fake - real.mean(), torch.ones_like(d_fake)))


def ragan_d(d_real, d_fake):
    bce = F.binary_cross_entropy_with_logits
    return 0.5 * (bce(d_real - d_fake.mean().detach(), torch.ones_like(d_real))
                  + bce(d_fake - d_real.mean().detach(), torch.zeros_like(d_fake)))


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


class Criterion:
    """Generator objective. Holds VGG and the discriminator(s)' G-side."""

    def __init__(self, cfg, device, vgg=None):
        self.per_map = cfg["per_map"]
        self.extra_l1 = cfg.get("extra_l1", {}) or {}
        self.joint_gan = float(cfg.get("joint_gan", 0.0))
        self.vgg = vgg
        self.render = None

    def pixel_terms(self, m, sr, hr, psnr_stage):
        spec, logs, total = self.per_map.get(m, {}), {}, 0.0
        if "l1" in spec:
            v = F.l1_loss(sr.float(), hr)
            w = 1.0 if psnr_stage else spec["l1"]
            total, logs["l1"] = total + w * v, v
        if "angular" in spec:
            if m != "normal":
                raise ValueError("angular loss only makes sense for normal maps")
            v = angular(sr, hr)
            w = 1.0 if psnr_stage else spec["angular"]
            total, logs["angular"] = total + w * v, v
        return total, logs

    def generator(self, sr, hr, discs, psnr_stage):
        total, logs = 0.0, {}
        for m in MAPS:
            t, lg = self.pixel_terms(m, sr[m], hr[m], psnr_stage)
            total = total + t
            logs.update({"%s/%s" % (m, k): v for k, v in lg.items()})
            if m in self.extra_l1:
                v = F.l1_loss(sr[m].float(), hr[m])
                total, logs["%s/extra_l1" % m] = total + self.extra_l1[m] * v, v
            if psnr_stage:
                continue
            spec = self.per_map.get(m, {})
            if spec.get("vgg", 0) > 0:
                v = self.vgg(sr[m].float(), hr[m])
                total, logs["%s/vgg" % m] = total + spec["vgg"] * v, v
            if spec.get("gan", 0) > 0 and discs.mode == "per_map":
                D = discs.for_map(m)
                v = ragan_g(D(hr[m]), D(sr[m]))
                total, logs["%s/gan_g" % m] = total + spec["gan"] * v, v
        if not psnr_stage and discs.mode == "joint" and self.joint_gan > 0:
            v = ragan_g(discs.joint(cat(hr)), discs.joint(cat(sr)))
            total, logs["joint/gan_g"] = total + self.joint_gan * v, v
        if self.render is not None and self.render.w > 0:
            v = self.render(sr, hr)
            total, logs["render"] = total + self.render.w * v, v
        return total, logs


def cat(maps):
    return torch.cat([maps[m] for m in MAPS], 1)


class Discriminators(torch.nn.Module):
    """mode: none | per_map (shared D over maps, or one per map when the
    generator is arch=separate) | joint (one D over the 15-ch concatenation)."""

    def __init__(self, mode, gan_maps, separate, make):
        super().__init__()
        self.mode, self.gan_maps, self.separate = mode, list(gan_maps), separate
        if mode == "per_map" and self.gan_maps:
            if separate:
                self.ds = torch.nn.ModuleDict({m: make(3) for m in self.gan_maps})
            else:
                self.shared = make(3)
        elif mode == "joint":
            self.joint = make(15)

    def active(self):
        return (self.mode == "per_map" and bool(self.gan_maps)) or self.mode == "joint"

    def for_map(self, m):
        return self.ds[m] if self.separate else self.shared

    def loss(self, sr, hr):
        sr = {m: v.detach() for m, v in sr.items()}
        if self.mode == "joint":
            return ragan_d(self.joint(cat(hr)), self.joint(cat(sr)))
        return sum(ragan_d(self.for_map(m)(hr[m]), self.for_map(m)(sr[m])) for m in self.gan_maps)
