"""
ESRGAN for SVBRDF maps (experiments 2 and 3).

RRDBNet / VGG-style discriminator follow BasicSR's definitions and parameter
names exactly, so the public RRDBNet checkpoints load without remapping
(RealESRGAN_x4plus: params_ema, 23 blocks, 64 feat, 32 grow). The original
ESRGAN repo's naming (RRDB_trunk / trunk_conv / upconv / HRconv) is remapped.

Map encoding: every map becomes a 3-channel image in [0,1]
    basecolor  sRGB as is
    normal     (n + 1) / 2, OpenGL
    roughness / metallic / height   grey replicated to 3 channels
Decoding reverses it: normals renormalised, grey maps = channel mean.

MultiMapSR arrangements (실험 2-2 / 2-4):
    input=independent, arch=shared    one 3->3 net, all maps through it (Alonso & Patow style)
    input=independent, arch=separate  one 3->3 net per map
    input=concat                      one 15->15 net over all maps at once
"""
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

MAPS = ["basecolor", "normal", "roughness", "metallic", "height"]
PRETRAINED_URLS = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
}


def _init(modules, scale=0.1):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        _init([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5])

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        # .float(): torch 2.0 has no bf16 CUDA kernel for nearest upsampling
        # ("upsample_nearest2d_out_frame not implemented for 'BFloat16'");
        # autocast casts back down for the conv that follows.
        up = lambda t: F.interpolate(t.float() if t.dtype == torch.bfloat16 else t,
                                     scale_factor=2, mode="nearest")
        feat = self.lrelu(self.conv_up1(up(feat)))
        feat = self.lrelu(self.conv_up2(up(feat)))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class VGGStyleDiscriminator(nn.Module):
    """BasicSR VGGStyleDiscriminator, 128x128 input."""

    def __init__(self, num_in_ch=3, nf=64):
        super().__init__()
        c = lambda i, o, k, s, b=True: nn.Conv2d(i, o, k, s, 1, bias=b)
        self.conv0_0, self.conv0_1, self.bn0_1 = c(num_in_ch, nf, 3, 1), c(nf, nf, 4, 2, False), nn.BatchNorm2d(nf, affine=True)
        self.conv1_0, self.bn1_0 = c(nf, nf * 2, 3, 1, False), nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1, self.bn1_1 = c(nf * 2, nf * 2, 4, 2, False), nn.BatchNorm2d(nf * 2, affine=True)
        self.conv2_0, self.bn2_0 = c(nf * 2, nf * 4, 3, 1, False), nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1, self.bn2_1 = c(nf * 4, nf * 4, 4, 2, False), nn.BatchNorm2d(nf * 4, affine=True)
        self.conv3_0, self.bn3_0 = c(nf * 4, nf * 8, 3, 1, False), nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1, self.bn3_1 = c(nf * 8, nf * 8, 4, 2, False), nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_0, self.bn4_0 = c(nf * 8, nf * 8, 3, 1, False), nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1, self.bn4_1 = c(nf * 8, nf * 8, 4, 2, False), nn.BatchNorm2d(nf * 8, affine=True)
        self.linear1 = nn.Linear(nf * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if x.shape[-1] != 128:
            raise ValueError("discriminator expects 128x128, got %s" % (tuple(x.shape),))
        L = self.lrelu
        f = L(self.conv0_0(x))
        f = L(self.bn0_1(self.conv0_1(f)))
        f = L(self.bn1_0(self.conv1_0(f)))
        f = L(self.bn1_1(self.conv1_1(f)))
        f = L(self.bn2_0(self.conv2_0(f)))
        f = L(self.bn2_1(self.conv2_1(f)))
        f = L(self.bn3_0(self.conv3_0(f)))
        f = L(self.bn3_1(self.conv3_1(f)))
        f = L(self.bn4_0(self.conv4_0(f)))
        f = L(self.bn4_1(self.conv4_1(f)))
        return self.linear2(L(self.linear1(f.flatten(1))))


class VGGPerceptual(nn.Module):
    """ESRGAN perceptual loss: VGG19 conv5_4 BEFORE activation, L1, ImageNet norm."""

    def __init__(self):
        super().__init__()
        import torchvision
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:35].eval()        # [34] = conv5_4
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        f = lambda t: self.features((t - self.mean) / self.std)
        return F.l1_loss(f(x), f(y.detach()))


class MultiMapSR(nn.Module):
    def __init__(self, arch="shared", input="independent", num_block=23, num_feat=64, num_grow_ch=32):
        super().__init__()
        if input not in ("independent", "concat") or arch not in ("shared", "separate"):
            raise ValueError((arch, input))
        if input == "concat" and arch == "separate":
            raise ValueError("concat input means one network over all maps; arch must be shared")
        self.arch, self.input = arch, input
        mk = lambda i, o: RRDBNet(i, o, num_feat, num_block, num_grow_ch)
        if input == "concat":
            self.net = mk(15, 15)
        elif arch == "shared":
            self.net = mk(3, 3)
        else:
            self.nets = nn.ModuleDict({m: mk(3, 3) for m in MAPS})

    def subnets(self):
        return list(self.nets.values()) if self.arch == "separate" else [self.net]

    def forward(self, lr):
        if self.input == "concat":
            y = self.net(torch.cat([lr[m] for m in MAPS], 1))
            return dict(zip(MAPS, torch.split(y, 3, dim=1)))
        if self.arch == "shared":
            y = self.net(torch.cat([lr[m] for m in MAPS], 0))
            return dict(zip(MAPS, torch.chunk(y, len(MAPS), dim=0)))
        return {m: self.nets[m](lr[m]) for m in MAPS}


# ------------------------------------------------------------ pretrained

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


def count_params(m):
    return sum(p.numel() for p in m.parameters())
