# DeepTextures - ESRGAN for texture super-resolution (Scientific Reports 2026)
# https://github.com/NethamGH/DeepTextures  @ e4ee562
#
# The base image ships TF 2.10.1 + CUDA 11.2 + cuDNN 8.1 on Python 3.8.
# CUDA 11.2 supports Ampere (sm_86 / RTX 3080 Ti).
#
# See requirements.txt for why the TF version cannot follow upstream's pin.
FROM tensorflow/tensorflow:2.10.1-gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# libgl1 + libglib2.0-0: required by opencv-python (libGL.so.1). Kept rather
# than switching to opencv-python-headless so the package list stays identical
# to upstream's.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Bake the VGG19 ImageNet weights (~80 MB) into the image.
# modules/losses.py builds VGG19(weights='imagenet') for the perceptual loss
# with this exact input_shape; upstream would download it from
# storage.googleapis.com on the first training step, making training depend on
# network access and on that URL still being alive.
RUN python -c "\
from tensorflow.keras.applications.vgg19 import VGG19; \
VGG19(input_shape=(None, None, 3), include_top=False); \
print('VGG19 imagenet weights cached')"

RUN python -c "\
import cv2, yaml, tqdm, PIL, tensorflow as tf; \
print('deeptex deps OK | TF', tf.__version__)"

WORKDIR /workspace/external/DeepTextures
CMD ["bash"]
