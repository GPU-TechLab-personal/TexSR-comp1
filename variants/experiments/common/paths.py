"""Container-side paths. Everything is fixed by experiments/docker/compose.yaml
(or the `docker run` mounts in variants/experiments/README.md).

Moved from experiments/common/paths.py. The names are the same; what changed:
  * WORKSPACE / MATUP come from variants/common/paths.py (REPO, MATUP_ROOT) -
    in the container both are the old literals, /workspace and
    /workspace/external/matup_egsr.
  * add_matup / add_swinir no longer edit sys.path themselves; they call
    variants/common/paths.py, the one place a submodule goes onto sys.path
    (variants/README.md rule 2).
"""
import os

from variants.common import paths as _root

WORKSPACE = _root.REPO
MATUP = _root.MATUP_ROOT
SWINIR_CKPT = os.path.join(
    MATUP, "pretrained_models", "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")

SHARDS = os.environ.get("MATSYNTH_SHARDS", "/data/MatSynth/data")
EXPDATA = os.environ.get("EXPDATA", "/expdata")

MATERIALS = os.path.join(EXPDATA, "materials")      # <set>/<name>/{LR,GT}/*.png
SR_CACHE = os.path.join(EXPDATA, "cache", "sr")     # <set>/<name>/radiance_*
PRETRAINED = os.path.join(EXPDATA, "cache", "pretrained")
RUNS = os.path.join(EXPDATA, "runs")                # exp1/, exp2/, ...

# The 12 evaluation materials (docs: texsr comparison scope) keep their
# provenance in runs/matsynth/<name>/meta.json.
EVAL12_META = os.path.join(WORKSPACE, "runs", "matsynth")


def add_matup():
    """Import MatUp's modules from outside the submodule, as matup_render.py does."""
    _root.use_matup()


def add_swinir():
    _root.use_swinir()
