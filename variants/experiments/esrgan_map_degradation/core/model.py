"""
ESRGAN building blocks for SVBRDF maps (experiments 2 and 3), shared by every arm.

RRDBNet / VGG-style discriminator follow BasicSR's definitions and parameter
names exactly, so the public RRDBNet checkpoints load without remapping
(RealESRGAN_x4plus: params_ema, 23 blocks, 64 feat, 32 grow). Loading them is
arm 2-1's code: arms/pretrained.py.

Map encoding: every map becomes a 3-channel image in [0,1]
    basecolor  sRGB as is
    normal     (n + 1) / 2, OpenGL
    roughness / metallic / height   grey replicated to 3 channels
Decoding reverses it: normals renormalised, grey maps = channel mean
(core/dataset.py).

How the maps are arranged around RRDBNet is the arm's choice, not this file's
(experiments/exp2_esrgan/model.py:MultiMapSR is split into):
    arms/base.py       SharedSR    input=independent, arch=shared   (2-0)
    arms/separate.py   SeparateSR  input=independent, arch=separate (2-2)
    arms/concat.py     ConcatSR    input=concat                     (2-4, 3-x)
core/assemble.py picks one from the config.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MAPS = ["basecolor", "normal", "roughness", "metallic", "height"]


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


def count_params(m):
    return sum(p.numel() for p in m.parameters())
