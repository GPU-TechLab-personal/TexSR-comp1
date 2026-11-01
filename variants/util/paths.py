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
    # official SR code (sr_finetune/): use_basicsr() / use_kair() / use_swinir_official()

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
# The official ESRGAN / SwinIR code sr_finetune/ trains and runs (never pip-installed).
BASICSR_ROOT = os.path.join(EXTERNAL, "BasicSR")
KAIR_ROOT = os.path.join(EXTERNAL, "KAIR")
SWINIR_ROOT = os.path.join(EXTERNAL, "SwinIR")
MATUP_SWINIR_ROOT = os.path.join(MATUP_ROOT, "sr_modules", "SwinIR")

# Commits docs/DEEPTEX-DEVIATIONS.md and docs/RUNBOOK.md cite line numbers from;
# for the last three, the commits sr_finetune/README.md calls "the official code".
PINS = {
    "DeepTextures": (DEEPTEX_ROOT, "e4ee562"),
    "matup_egsr": (MATUP_ROOT, "c74ec99"),
    "BasicSR": (BASICSR_ROOT, "651835a"),   # tag v1.4.2
    "KAIR": (KAIR_ROOT, "fc1732f"),
    "SwinIR": (SWINIR_ROOT, "6545850"),     # the commit MatUp's sr_modules/SwinIR copy is at
}
BASICSR_VERSION = ("1.4.2", "651835a1b9d38dbbdaf45750f56906be2364f01a")


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


# Each of these ships the top-level packages `models` / `utils` (KAIR also `data`,
# which MatUp's root has too), so only one of them can be on sys.path in a process:
# with two, `import models.network_swinir` silently picks whichever is first.
_TOPLEVEL = {"KAIR": KAIR_ROOT, "SwinIR": SWINIR_ROOT,
             "matup SwinIR copy": MATUP_SWINIR_ROOT, "matup": MATUP_ROOT}


def _claim_toplevel(owner):
    root = _TOPLEVEL[owner]
    others = [n for n, p in _TOPLEVEL.items() if n != owner and p in sys.path]
    for pkg in ("models", "utils", "data"):
        mod = sys.modules.get(pkg)
        where = getattr(mod, "__file__", None) or next(iter(getattr(mod, "__path__", [])), None)
        if where and not os.path.abspath(where).startswith(root + os.sep):
            others.append("`%s` already imported from %s" % (pkg, where))
    if others:
        raise SystemExit("[fatal] paths.use_%s: %s is on sys.path / imported too; "
                         "its top-level packages would mix with %s's. Use one per process."
                         % (owner.lower(), ", ".join(others), owner))
    _front(root)
    return root


def use_basicsr():
    """`import basicsr` resolves to external/BasicSR/basicsr.

    The source tree has no basicsr/version.py (setup.py writes it), and
    basicsr/__init__.py ends with `from .version import ...`; a module with the
    pinned version is put in sys.modules instead, so nothing is written into the
    submodule. The CUDA ops are not built: ESRGAN uses none of them."""
    import types
    ver = types.ModuleType("basicsr.version")
    ver.__version__, ver.__gitsha__ = BASICSR_VERSION
    ver.version_info = tuple(int(x) for x in BASICSR_VERSION[0].split("."))
    sys.modules.setdefault("basicsr.version", ver)
    _front(BASICSR_ROOT)
    _warn_official_pin("BasicSR")
    return BASICSR_ROOT


def use_kair():
    """`import models.* / utils.* / data.*` resolve to external/KAIR."""
    root = _claim_toplevel("KAIR")
    _warn_official_pin("KAIR")
    return root


def use_swinir_official():
    """`import models.network_swinir` resolves to external/SwinIR (the release the
    official weights belong to), not MatUp's copy (use_swinir)."""
    root = _claim_toplevel("SwinIR")
    _warn_official_pin("SwinIR")
    return root


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


def _warn_official_pin(name):
    """check_pin for the official SR code: a different commit is no longer the
    code sr_finetune/README.md says was run. Never fatal; gate env is strict."""
    root, pinned = PINS[name]
    head = submodule_head(root)
    if head is not None and not head.startswith(pinned):
        print("! external/%s is at %s, not the pinned %s: not the official code "
              "sr_finetune/README.md describes." % (name, head[:12], pinned))
