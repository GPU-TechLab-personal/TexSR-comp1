# The project's shared torch environment: torch 2.0.0 / CUDA 11.7 plus the common
# image, numerics and logging packages, every version pinned. New torch
# experiments build on this (or run on it directly, like the srft service) instead
# of on a method's image.
#
# It replaces nothing: texsr/matup and texsr/exp stay as they are, because they
# exist to freeze MatUp's environment (docs/RUNBOOK.md section 8, rule 1) and
# rebuilding them would re-resolve their unpinned requirements. The pins here ARE
# that environment, measured: constraints.txt is every package texsr/matup
# installs or changes on top of the same base image, at the version it has
# (pip freeze diff, 2026-11-01), with scipy the one deliberate change.
#
# TensorFlow (texsr/deeptex, CUDA 11.2) cannot share this base; texsr/dataprep
# has no CUDA at all.
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# PYTHONDONTWRITEBYTECODE: code runs from the /workspace mount, submodules
# included; importing it must not leave __pycache__ there.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Same three as docker/matup: git (paths.submodule_head), libgl1 + libglib2.0-0
# for opencv-python's libGL.so.1.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt constraints.txt check_pins.py /tmp/torchbase/
RUN pip install -r /tmp/torchbase/requirements.txt -c /tmp/torchbase/constraints.txt

# Fail the BUILD, not the first run: every installed package named in
# constraints.txt is at its pinned version, torch is the base image's, and no
# installed package owns a name the submodules provide (basicsr, models, utils,
# data - variants/util/paths.py), or sys.path order would decide which code runs.
RUN python -I /tmp/torchbase/check_pins.py /tmp/torchbase/constraints.txt

WORKDIR /workspace
CMD ["bash"]
