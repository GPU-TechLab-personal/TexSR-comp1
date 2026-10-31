"""Container-side paths. Everything is fixed by experiments/docker/compose.yaml."""
import os
import sys

WORKSPACE = "/workspace"
MATUP = os.path.join(WORKSPACE, "external", "matup_egsr")
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
    if MATUP not in sys.path:
        sys.path.insert(0, MATUP)


def add_swinir():
    add_matup()
    p = os.path.join(MATUP, "sr_modules", "SwinIR")
    if p not in sys.path:
        sys.path.insert(0, p)
