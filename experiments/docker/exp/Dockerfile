# Experiment image for experiments/ (실험 1-5).
#
# Built ON TOP of texsr/matup rather than from a fresh torch base, on purpose:
# every experiment drives MatUp's own renderer, dataset and SwinIR hookup
# (external/matup_egsr, imported from outside, never edited). Inheriting the
# exact torch 2.0.0 / CUDA 11.7 stack means a MatUp run inside this image is
# the same computation as one in the matup image - the only way the
# "MatUp original" arm of experiment 4 stays the original.
#
# The matup image itself is NOT modified (docs/RUNBOOK.md section 8, rule 1):
# everything extra lands in this derived image only.
#
# Build the base first if it is missing:
#     cd docker; docker compose build matup
FROM texsr/matup:torch2.0.0-cu117

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/expdata/cache/huggingface \
    TORCH_HOME=/expdata/cache/torch

COPY requirements.txt /tmp/requirements-exp.txt
RUN pip install -r /tmp/requirements-exp.txt

# Fail the BUILD, not the first experiment. torch must still be the matup pin:
# a transitive dependency silently upgrading it would break the premise above.
RUN python -c "\
import torch, torchvision, pyarrow, lpips, skimage, flip_evaluator, diffusers, transformers; \
from diffusers import StableDiffusionUpscalePipeline, AutoencoderKL, DiTTransformer2DModel; \
from timm.models.layers import DropPath; \
assert torch.__version__.startswith('2.0.0'), torch.__version__; \
print('exp deps OK | torch', torch.__version__, '| diffusers', diffusers.__version__, \
      '| transformers', transformers.__version__, '| pyarrow', pyarrow.__version__)"

WORKDIR /workspace
CMD ["bash"]
