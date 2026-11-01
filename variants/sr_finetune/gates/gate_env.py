#!/usr/bin/env python3
"""Gate env: the official SR code is the pinned code, and the srft service
(texsr/torchbase, docker/torchbase) runs it.

Two parts, because each question has one place it can be answered honestly:

  --part host    on the host (plain python, stdlib only), where git sees the
                 checkout as it really is:
                     python variants/sr_finetune/gates/gate_env.py --part host
                 -> runs/sr_ft/gate/env_host.json
  --part image   in the srft service (texsr/torchbase):
                     docker compose run --rm srft python variants/sr_finetune/gates/gate_env.py --part image
                 -> runs/sr_ft/gate/env_image.json

host
  pins        each of external/{BasicSR,KAIR,SwinIR} has HEAD at paths.PINS
  clean       `git status --porcelain` is empty in each (nothing written into a
              submodule; run it again after the other gates). The container's
              git would see a Windows checkout's line endings as edits, so this
              is not asked there.
  swinir_src  sha256 of models/network_swinir.py in KAIR, external/SwinIR and
              MatUp's copy. RECORDED, not judged: write_arm.py (step 4) loads a
              KAIR-trained checkpoint into KAIR's file if KAIR's differs.

image
  versions    torch 2.0.0, scipy 1.11.4 (interp2d works when CALLED, scipy.finfo exists),
              numpy 1.x, CUDA visible
  no_shadow   no installed package named basicsr / models / utils / data (asked by
              a `python -I` from /, so only site-packages can answer)
  imports     one subprocess per code base, because paths.use_kair() and
              use_swinir_official() refuse to share a process; every module's
              __file__ must lie inside its submodule. basicsr is imported WHOLE
              (`import basicsr` loads every arch/data/loss/metric module), so a
              package the image lacks fails here.
  bsrgan      KAIR's degradation_bsrgan on one MatSynth HR crop, as
              data/dataset_blindsr.py:68-70 calls it: 256^2 -> LQ 64^2 / HQ 256^2,
              finite, in [0,1]

"pass" is what Show-GateResults (util/lib.ps1) reads. Exit 0 on pass, 1 on fail.
"""
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.util import paths  # noqa: E402
from variants.sr_finetune.gates import probe as child  # noqa: E402

GATE_DIR = os.path.join(paths.REPO, "runs", "sr_ft", "gate")
OFFICIAL = ("BasicSR", "KAIR", "SwinIR")
HR_DIR = os.path.join(paths.REPO, "runs", "deeptex", "datasets", "matsynth", "baseColor", "HR")

# What each code base must import from its own submodule: the modules the
# training and inference paths of sr_finetune/ use.
PROBES = {
    "basicsr": ("use_basicsr", paths.BASICSR_ROOT, [
        "basicsr", "basicsr.train", "basicsr.models.esrgan_model",
        "basicsr.archs.rrdbnet_arch", "basicsr.archs.discriminator_arch",
        "basicsr.archs.vgg_arch", "basicsr.losses.basic_loss", "basicsr.losses.gan_loss",
        "basicsr.data.paired_image_dataset"]),
    "kair": ("use_kair", paths.KAIR_ROOT, [
        "models.select_model", "models.model_plain", "models.model_gan",
        "models.select_network", "models.network_swinir", "models.network_discriminator",
        "models.loss", "data.select_dataset", "data.dataset_sr", "data.dataset_blindsr",
        "utils.utils_image", "utils.utils_blindsr", "utils.utils_option",
        "utils.utils_logger", "utils.utils_dist"]),
    "swinir": ("use_swinir_official", paths.SWINIR_ROOT, ["models.network_swinir"]),
}


def write_json(name, obj):
    os.makedirs(GATE_DIR, exist_ok=True)
    path = os.path.join(GATE_DIR, name)
    with open(path + ".partial", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(path + ".partial", path)
    return path


def sha256(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git(root, *args):
    try:
        return subprocess.check_output(["git", "-C", root] + list(args),
                                       stderr=subprocess.STDOUT).decode()
    except (OSError, subprocess.CalledProcessError) as e:
        return None if isinstance(e, OSError) else "ERROR " + e.output.decode()


def part_host():
    checks = {}
    pins = {}
    for name in OFFICIAL:
        root, pinned = paths.PINS[name]
        head = paths.submodule_head(root)
        pins[name] = {"root": os.path.relpath(root, paths.REPO), "pinned": pinned,
                      "head": head, "ok": bool(head and head.startswith(pinned))}
    checks["pins"] = {"ok": all(p["ok"] for p in pins.values()), "submodules": pins}

    clean = {}
    for name in OFFICIAL:
        out = git(paths.PINS[name][0], "status", "--porcelain", "--untracked-files=all")
        clean[name] = {"ok": out == "", "status": out}
    checks["clean"] = {"ok": all(c["ok"] for c in clean.values()), "submodules": clean}

    src = {"KAIR": os.path.join(paths.KAIR_ROOT, "models", "network_swinir.py"),
           "SwinIR": os.path.join(paths.SWINIR_ROOT, "models", "network_swinir.py"),
           "matup SwinIR copy": os.path.join(paths.MATUP_SWINIR_ROOT, "models", "network_swinir.py")}
    shas = {k: sha256(v) for k, v in src.items()}
    checks["swinir_src"] = {"ok": all(shas.values()), "sha256": shas,
                            "kair_equals_official": shas["KAIR"] == shas["SwinIR"],
                            "official_equals_matup_copy": shas["SwinIR"] == shas["matup SwinIR copy"]}
    return checks


def probe(which):
    """Child process: put one code base on sys.path, import its modules, report."""
    use, root, modules = PROBES[which]
    getattr(paths, use)()
    out = {"modules": {}}
    for m in modules:
        try:
            mod = importlib.import_module(m)
            f = os.path.abspath(mod.__file__)
            out["modules"][m] = {"file": os.path.relpath(f, paths.REPO),
                                 "inside": f.startswith(root + os.sep)}
        except Exception as e:  # noqa: BLE001 - the point is to report it
            out["modules"][m] = {"error": "%s: %s" % (type(e).__name__, e), "inside": False}
    out["ok"] = all(v["inside"] for v in out["modules"].values())
    if which == "basicsr" and out["ok"]:
        import basicsr
        out["version"] = [basicsr.__version__, basicsr.__gitsha__]
        out["ok"] = basicsr.__version__ == paths.BASICSR_VERSION[0]
    if which == "kair" and out["ok"]:
        try:
            out["bsrgan"] = bsrgan()
        except Exception as e:  # noqa: BLE001 - e.g. a scipy name the image lacks
            out["bsrgan"] = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
        out["ok"] = out["bsrgan"]["ok"]
    child.emit(out)


def bsrgan():
    import random
    import numpy as np
    from utils import utils_blindsr as blindsr
    from utils import utils_image as util
    name = sorted(f for f in os.listdir(HR_DIR) if f.endswith(".png"))[0]
    img = util.uint2single(util.imread_uint(os.path.join(HR_DIR, name), 3)[:256, :256])
    random.seed(0)
    np.random.seed(0)
    lq, hq = blindsr.degradation_bsrgan(img, 4, lq_patchsize=64, isp_model=None)
    res = {"input": name, "lq_shape": list(lq.shape), "hq_shape": list(hq.shape),
           "finite": bool(np.isfinite(lq).all() and np.isfinite(hq).all()),
           "lq_range": [float(lq.min()), float(lq.max())]}
    res["ok"] = (res["lq_shape"] == [64, 64, 3] and res["hq_shape"] == [256, 256, 3]
                 and res["finite"] and -1e-6 <= res["lq_range"][0] and res["lq_range"][1] <= 1 + 1e-6)
    return res


def part_image():
    import numpy as np
    import scipy
    import torch
    checks = {}
    # KAIR utils_blindsr.py:12. Called, not imported: from scipy 1.14 on the name
    # imports fine and raises NotImplementedError when used.
    try:
        from scipy.interpolate import interp2d
        interp2d(np.arange(3.0), np.arange(3.0), np.zeros((3, 3)))(0.5, 0.5)
        has_interp2d = True
    except Exception:  # noqa: BLE001
        has_interp2d = False
    v = {"torch": torch.__version__, "scipy": scipy.__version__, "numpy": np.__version__,
         "interp2d": has_interp2d, "cuda": torch.cuda.is_available(),
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    v["finfo"] = hasattr(scipy, "finfo")   # KAIR utils_blindsr.py:195
    v["ok"] = (v["torch"].startswith("2.0.0") and v["scipy"] == "1.11.4" and v["finfo"]
               and v["numpy"].startswith("1.") and has_interp2d and v["cuda"])
    checks["versions"] = v

    code = ("import importlib.util as u, json; print(json.dumps({n: u.find_spec(n) is not None "
            "for n in ('basicsr', 'models', 'utils', 'data')}))")
    found = json.loads(subprocess.check_output([sys.executable, "-I", "-c", code], cwd="/"))
    checks["no_shadow"] = {"ok": not any(found.values()), "installed": found}

    imports = {}
    for which in PROBES:
        imports[which] = child.run(os.path.abspath(__file__), which)
    checks["imports"] = {"ok": all(i["ok"] for i in imports.values()), "code_bases": imports}
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["host", "image"])
    ap.add_argument("--probe", choices=sorted(PROBES), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.probe:
        return probe(args.probe)
    if not args.part:
        ap.error("--part host|image")

    checks = part_host() if args.part == "host" else part_image()
    report = {"part": args.part, "checks": checks,
              "failed": [k for k, c in checks.items() if not c["ok"]]}
    report["pass"] = not report["failed"]
    path = write_json("env_%s.json" % args.part, report)
    for k, c in checks.items():
        print("  %-11s %s" % (k, "ok" if c["ok"] else "FAIL"))
    print("[gate env/%s] %s -> %s" % (args.part, "PASS" if report["pass"] else "FAIL", path))
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
