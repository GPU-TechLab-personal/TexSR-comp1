"""Where everything is, and the one place the submodules go onto sys.path.

The old tree did this file by file: deeptex_train.py:82-94, deeptex_compare.py:78-82,
render_calibrate.py:96-100, render_gate.py:131-133, matup_render.py:40,
experiments/common/paths.py:22-33 ... each with its own copy of the
DEEPTEX_ROOT default and, in deeptex_train.py:111-124, the only pin check.

Every entry point in variants/ starts with

    import os, sys
    sys.path.insert(0, <repo root>)          # so `import variants` works
    from variants.util import paths
    paths.use_deeptex()                      # or use_matup() / use_donor()

and imports nothing from scripts/ or experiments/. The submodules are the only
outside code variants/ may import (variants/README.md).

Container layout (docker/compose.yaml, docker/exp/compose.yaml):
/workspace is the repo. Host runs (the L0 checker, the L5 harness) resolve the
repo from this file's location instead.
"""

import os
import subprocess
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir))
VARIANTS = os.path.join(REPO, "variants")
EXTERNAL = os.path.join(REPO, "external")

# The env overrides are the old tree's (deeptex_train.py:82, deeptex_sr.py:63, ...).
DEEPTEX_ROOT = os.environ.get("DEEPTEX_ROOT", os.path.join(EXTERNAL, "DeepTextures"))
MATUP_ROOT = os.path.join(EXTERNAL, "matup_egsr")
DONOR_ROOT = os.path.join(EXTERNAL, "esrgan-tf2")

# Commits docs/DEEPTEX-DEVIATIONS.md and docs/RUNBOOK.md cite line numbers from.
PINS = {
    "DeepTextures": (DEEPTEX_ROOT, "e4ee562"),
    "matup_egsr": (MATUP_ROOT, "c74ec99"),
}


def _front(p):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


def use_deeptex():
    """`import modules.*` resolves to external/DeepTextures/modules.

    esrgan-tf2 has a `modules` package too; the donor is never put on sys.path
    (psnr_loop.py drives DeepTextures' modules, as deeptex_train.py --stage psnr did)."""
    _front(DEEPTEX_ROOT)
    return DEEPTEX_ROOT


def use_matup():
    """`import matup.*` resolves to external/matup_egsr (matup_render.py:40)."""
    _front(MATUP_ROOT)
    return MATUP_ROOT


def use_swinir():
    use_matup()
    _front(os.path.join(MATUP_ROOT, "sr_modules", "SwinIR"))


def submodule_head(root):
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001 - git may be absent inside the container
        return None


def check_pin(name="DeepTextures"):
    """deeptex_train.py:111-124's warning, for either submodule. Never fatal: the
    run still means something, only the cited line numbers may be off."""
    root, pinned = PINS[name]
    head = submodule_head(root)
    if head is None:
        return
    if not head.startswith(pinned):
        print("!" * 72)
        print("! external/%s is at %s, not the pinned %s." % (name, head[:12], pinned))
        print("! docs/DEEPTEX-DEVIATIONS.md and docs/RUNBOOK.md 4 cite line numbers")
        print("! from the pin and may now point at the wrong lines.")
        print("!" * 72)
