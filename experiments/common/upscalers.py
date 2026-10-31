"""
Image upsamplers S (실험 1의 변수 S, 실험 4에서 동일하게 재사용).

    get("swinir")   SwinIR-L x4 GAN (real_sr, the checkpoint MatUp ships with)
    get("srflow")   SRFlow x4, heat t=0.5. The checkpoint is no longer
                    distributed (README); this adapter runs only if the
                    SRFlow code and checkpoint are supplied by hand.
    get("sdx4")     diffusion: stabilityai/stable-diffusion-x4-upscaler
    get("bilinear") / get("bicubic")   controls; C is ~0 for bilinear by construction

All share one call: up(x) with x an (N, 3, h, w) float tensor, RGB, [0,1],
returning (N, 3, 4h, 4w) in [0,1], rounded to the 8-bit ladder the way
sr_swinir.py saves it. The INPUT must already be what MatUp's upsampler
reads - sr_cache.read_lr_renders produces exactly that (16-bit PNG ->
cv2.IMREAD_COLOR -> /255). It is deliberately not re-quantised here: on the
GPU, round(x*255)/255 is computed with a reciprocal multiply and moves k/255
by one ulp, which flipped ~1% of SwinIR's output roundings against MatUp's
own run (measured, experiments/gates/gate_matup_driver.py).

`tag` identifies S *with its settings* - two diffusion runs with different
step counts must never share a cache directory.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from . import paths


class UpscalerUnavailable(RuntimeError):
    pass


def quantize8(x):
    return torch.round(torch.clamp(x, 0, 1) * 255.0) / 255.0


class _Base:
    scale = 4
    deterministic = True

    def config(self):
        return {"name": self.name, "tag": self.tag}

    def __call__(self, x):
        return quantize8(self.up(x))


class Interp(_Base):
    def __init__(self, mode, device):
        self.name = self.tag = mode
        self.mode = mode

    def up(self, x):
        return torch.clamp(F.interpolate(x, scale_factor=4, mode=self.mode,
                                         align_corners=False), 0, 1)


def bilinear(x):
    """The bilinear reference of experiment 1's D. cv2.INTER_LINEAR and this
    agree for upsampling (half-pixel centres, edge replicate)."""
    return F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)


class SwinIR(_Base):
    """Mirrors sr_swinir.py:main_upsample_swinir: same model constructor, same
    mirror padding to (h // 8 + 1) * 8, same grouping into up_batch_size
    forward passes."""
    name = "swinir"

    def __init__(self, device, up_batch_size=10, ckpt=paths.SWINIR_CKPT):
        import argparse
        if not os.path.isfile(ckpt):
            raise UpscalerUnavailable("SwinIR checkpoint missing: %s (README step 3)" % ckpt)
        paths.add_swinir()
        cwd = os.getcwd()
        try:
            os.chdir(paths.MATUP)          # main_test_swinir resolves relative imports from here
            from sr_modules.SwinIR.main_test_swinir import define_model, test
        finally:
            os.chdir(cwd)
        self._test = test
        self.args = argparse.Namespace(task="real_sr", scale=4, large_model=True,
                                       model_path=ckpt, tile=None, tile_overlap=32,
                                       folder_gt=None)
        self.model = define_model(self.args).eval().to(device)
        self.window = 8
        self.up_batch_size = up_batch_size
        self.tag = "swinir"

    @torch.no_grad()
    def up(self, x):
        n, _, h, w = x.shape
        hp = (h // self.window + 1) * self.window - h
        wp = (w // self.window + 1) * self.window - w
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h + hp, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w + wp]
        per = max(1, n // self.up_batch_size) if n >= self.up_batch_size else n
        outs = []
        for i in range(0, n, per):
            o = self._test(x[i:i + per], self.model, self.args, self.window)
            outs.append(o[..., :h * 4, :w * 4])
        return torch.clamp(torch.cat(outs), 0, 1)


class SRFlow(_Base):
    name = "srflow"
    deterministic = False

    def __init__(self, device, heat=0.5, seed=0, up_batch_size=10,
                 code_dir=os.path.join(paths.MATUP, "sr_modules", "SRFlow", "code"),
                 conf=os.path.join(paths.MATUP, "sr_modules", "SRFlow_DF2K_4X_custom.yml")):
        if not os.path.isdir(code_dir) or not os.listdir(code_dir):
            raise UpscalerUnavailable(
                "SRFlow code not present at %s. The submodule is left empty on purpose: "
                "the checkpoint is no longer distributed and the licence is CC BY-NC-SA "
                "(README step 2). Check it out and place the checkpoint the conf names, "
                "then pass --s srflow again." % code_dir)
        sys.path.append(code_dir)
        cwd = os.getcwd()
        try:
            os.chdir(paths.MATUP)
            from test import load_model     # SRFlow/code/test.py, as sr_srflow.py does
            self.model, _ = load_model(conf)
        finally:
            os.chdir(cwd)
        self.heat, self.seed, self.up_batch_size = heat, seed, up_batch_size
        self.tag = "srflow_t%g_s%d" % (heat, seed)

    @torch.no_grad()
    def up(self, x):
        torch.manual_seed(self.seed)
        outs = []
        for i in range(0, x.shape[0], self.up_batch_size):
            outs.append(self.model.get_sr(lq=x[i:i + self.up_batch_size].cpu(),
                                          heat=self.heat).to(x.device))
        return torch.clamp(torch.cat(outs), 0, 1)

    def config(self):
        return {"name": self.name, "tag": self.tag, "heat": self.heat, "seed": self.seed}


class SDx4(_Base):
    """Stable Diffusion x4 upscaler. Stochastic: each light gets its own noise
    (seed_mode=per_light), like independent calls would; seed_mode=shared uses
    one noise draw for every light, which by itself lowers the cross-light
    inconsistency - a knob, recorded in the tag."""
    name = "sdx4"
    deterministic = False
    MODEL = "stabilityai/stable-diffusion-x4-upscaler"

    def __init__(self, device, steps=20, noise_level=20, guidance=0.0, seed=0,
                 seed_mode="per_light", batch=4, prompt=""):
        from diffusers import StableDiffusionUpscalePipeline
        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self.MODEL, torch_dtype=torch.float16).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device
        self.steps, self.noise_level, self.guidance = steps, noise_level, guidance
        self.seed, self.seed_mode, self.batch, self.prompt = seed, seed_mode, batch, prompt
        self.tag = "sdx4_s%d_n%d_g%g_%s%d" % (steps, noise_level, guidance, seed_mode, seed)
        self.counter = 0

    @torch.no_grad()
    def up(self, x):
        from PIL import Image
        outs = []
        for i in range(0, x.shape[0], self.batch):
            chunk = x[i:i + self.batch]
            imgs = [Image.fromarray((c.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
                    for c in chunk]
            gens = [torch.Generator(self.device).manual_seed(
                self.seed if self.seed_mode == "shared" else self.seed + self.counter + j)
                for j in range(len(imgs))]
            self.counter += len(imgs)
            o = self.pipe(prompt=[self.prompt] * len(imgs), image=imgs,
                          num_inference_steps=self.steps, noise_level=self.noise_level,
                          guidance_scale=self.guidance, generator=gens, output_type="np").images
            outs.append(torch.from_numpy(np.asarray(o)).permute(0, 3, 1, 2).float().to(x.device))
        return torch.clamp(torch.cat(outs), 0, 1)

    def config(self):
        return {"name": self.name, "tag": self.tag, "model": self.MODEL, "steps": self.steps,
                "noise_level": self.noise_level, "guidance": self.guidance,
                "seed": self.seed, "seed_mode": self.seed_mode, "prompt": self.prompt}


REGISTRY = {
    "bilinear": lambda device, **kw: Interp("bilinear", device),
    "bicubic": lambda device, **kw: Interp("bicubic", device),
    "swinir": lambda device, **kw: SwinIR(device, **kw),
    "srflow": lambda device, **kw: SRFlow(device, **kw),
    "sdx4": lambda device, **kw: SDx4(device, **kw),
}


def get(name, device, **kw):
    if name not in REGISTRY:
        raise KeyError("unknown upscaler %r (have %s)" % (name, sorted(REGISTRY)))
    return REGISTRY[name](device, **kw)


def parse_opts(items):
    """['steps=20', 'seed_mode=shared'] -> kwargs with numbers converted."""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        out[k] = v
    return out
