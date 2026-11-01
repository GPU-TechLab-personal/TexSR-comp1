"""The official networks and weights sr_finetune/ starts from - one table, read by
weights.py, the gates and write_arm.py, so an arm cannot be built one way in one
place and another way in the next.

ARCHS    constructor arguments copied from the official inference code (cited per
         entry). `code` is the code base the class comes from; build() needs that
         code base on sys.path first (paths.use_basicsr / use_swinir_official /
         use_kair - one per process).
WEIGHTS  the files, where they came from, what their sha256 must start with when
         the publisher put it in the name, and which key holds the state_dict.

Every torch.load here is weights_only=True: a .pth is a pickle, and these are
downloaded files.
"""
import importlib
import os

from variants.util import paths

WEIGHTS_DIR = os.path.join(paths.REPO, "runs", "sr_ft", "weights")
TORCH_HUB = os.path.join(paths.REPO, "runs", "sr_ft", "cache", "torch", "hub", "checkpoints")

_SWINIR_M = dict(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0,
                 depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                 mlp_ratio=2, resi_connection="1conv")

ARCHS = {
    # BasicSR inference/inference_esrgan.py:25
    "rrdb": {"code": "basicsr", "module": "basicsr.archs.rrdbnet_arch", "cls": "RRDBNet",
             "kwargs": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)},
    # SwinIR main_test_swinir.py:131-133, task classical_sr, --training_patch_size 64
    # (the "s64" in the weight's name). img_size matters: the attention-mask
    # buffers are in the state_dict at that size.
    "swinir_m_classical": {"code": "swinir", "module": "models.network_swinir", "cls": "SwinIR",
                           "kwargs": dict(_SWINIR_M, upsampler="pixelshuffle")},
    # main_test_swinir.py:148-150, task real_sr
    "swinir_m_real": {"code": "swinir", "module": "models.network_swinir", "cls": "SwinIR",
                      "kwargs": dict(_SWINIR_M, upsampler="nearest+conv")},
    # main_test_swinir.py:153-156, task real_sr --large_model
    "swinir_l_real": {"code": "swinir", "module": "models.network_swinir", "cls": "SwinIR",
                      "kwargs": dict(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0,
                                     depths=[6] * 9, embed_dim=240, num_heads=[8] * 9, mlp_ratio=2,
                                     upsampler="nearest+conv", resi_connection="3conv")},
}

SWINIR_RELEASE = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
WEIGHTS = {
    # BasicSR scripts/download_pretrained_models.py:40-41 (Google Drive id); the
    # name's -ff704c30 is the first 8 hex of its sha256 (BasicSR's convention).
    "esrgan": {"file": os.path.join(WEIGHTS_DIR, "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth"),
               "source": "https://drive.google.com/file/d/1b3_bWZTjNO3iL2js1yWkJfjZykcQgvzT",
               "sha_prefix": "ff704c30", "key": "params", "arch": "rrdb"},
    "swinir_cls": {"file": os.path.join(WEIGHTS_DIR, "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"),
                   "source": SWINIR_RELEASE + "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
                   "sha_prefix": None, "key": "params", "arch": "swinir_m_classical"},
    # Only params_ema was released (no non-EMA generator, no discriminator).
    "swinir_real": {"file": os.path.join(WEIGHTS_DIR, "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"),
                    "source": SWINIR_RELEASE + "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
                    "sha_prefix": None, "key": "params_ema", "arch": "swinir_m_real"},
    # Already in the MatUp submodule's pretrained_models/ (untracked there, put by
    # its download_links.txt); zero-shot only, read in place.
    "swinir_realL": {"file": os.path.join(paths.MATUP_ROOT, "pretrained_models",
                                          "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"),
                     "source": SWINIR_RELEASE + "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
                     "sha_prefix": None, "key": "params_ema", "arch": "swinir_l_real"},
    # The perceptual losses' VGG19 (BasicSR archs/vgg_arch.py and KAIR models/loss.py
    # both call torchvision's vgg19(pretrained=True), which reads $TORCH_HOME/hub/
    # checkpoints). -dcbb9e9d is again the sha256 prefix (torchvision's convention).
    "vgg19": {"file": os.path.join(TORCH_HUB, "vgg19-dcbb9e9d.pth"),
              "source": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
              "sha_prefix": "dcbb9e9d", "key": None, "arch": None},
}


# tag -> the official weight it starts from (its arch, its zero-shot file) and,
# for a fine-tuned arm, where its trainer keeps the inference weights:
#   BasicSR  net_g_<N>.pth = {'params', 'params_ema'} (SRModel.save with EMA); the EMA is inferred.
#   KAIR     <N>_E.pth, a bare state_dict of the EMA network netE (model_base.save_network).
# Inference always uses the OFFICIAL network file (BasicSR's RRDBNet, external/SwinIR's
# SwinIR); a KAIR-trained state_dict loads into it strictly (gate weights, kair_init).
ARMS = {
    "esrgan_zs":       {"weights": "esrgan"},
    "esrgan_ft":       {"weights": "esrgan", "ft_file": "net_g_{step}.pth", "ft_key": "params_ema"},
    "swinir_cls_zs":   {"weights": "swinir_cls"},
    "swinir_cls_ft":   {"weights": "swinir_cls", "ft_file": "{step}_E.pth", "ft_key": None},
    "swinir_real_zs":  {"weights": "swinir_real"},
    "swinir_real_ft":  {"weights": "swinir_real", "ft_file": "{step}_E.pth", "ft_key": None},
    "swinir_realL_zs": {"weights": "swinir_realL"},
}


def load_state(path, key):
    """The state_dict in `path` under `key` (None: the file is the state_dict)."""
    import torch
    d = torch.load(path, map_location="cpu", weights_only=True)
    if key is None:
        return d
    if key not in d:
        raise KeyError("%s has no '%s' (keys: %s)" % (path, key, sorted(d)[:5]))
    return d[key]


def build(arch):
    """A fresh network for ARCHS[arch]; its code base must already be on sys.path."""
    a = ARCHS[arch]
    return getattr(importlib.import_module(a["module"]), a["cls"])(**a["kwargs"])
