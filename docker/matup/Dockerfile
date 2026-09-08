# MatUp - "Repurposing Image Upsamplers for SVBRDFs" (EGSR 2024)
# https://github.com/AlbanGauthier/matup_egsr  @ c74ec99
#
# The base image provides exactly the torch 2.0.0 / CUDA 11.7 / cuDNN 8 stack
# that MatUp pins, and CUDA 11.7 supports Ampere (sm_86 / RTX 3080 Ti).
#
# MatUp's optimizer uses torch.cuda.CUDAGraph() and Adam(capturable=True), so
# this container REQUIRES an NVIDIA GPU - there is no CPU fallback path.
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# libgl1 + libglib2.0-0: required by opencv-python (libGL.so.1). Kept rather
# than switching to opencv-python-headless so the package list stays identical
# to upstream's.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Fail the BUILD rather than the first experiment if the two corrections in
# requirements.txt are wrong. The timm import below is verbatim what
# SwinIR/models/network_swinir.py performs.
RUN python -c "\
from timm.models.layers import DropPath, to_2tuple, trunc_normal_; \
import requests, cv2, lpips, torch, torchvision; \
print('matup deps OK | torch', torch.__version__, '| torchvision', torchvision.__version__)"

WORKDIR /workspace/external/matup_egsr
CMD ["bash"]
