#!/usr/bin/env python3
"""Gate weights: every official weight is the published file and fits the
official network exactly. Runs in the srft service:

    docker compose run --rm srft python variants/sr_finetune/gates/gate_weights.py

-> runs/sr_ft/gate/weights.json ("pass" is what Show-GateResults reads).

  manifest   weights.py --check: present, sha256 (publisher's prefix where the name
             has one), key -> runs/sr_ft/weights/manifest.json
  strict     each weight into nets.build(arch) with load_state_dict(strict=True),
             in a child process holding only that weight's code base (BasicSR for
             ESRGAN, external/SwinIR for the three SwinIR), then one forward on a
             1x3x32x32 input: output 1x3x128x128 and finite. Strict means no
             missing and no unexpected key; with SwinIR that includes the
             attention-mask buffers, so a wrong img_size fails here.
  kair_init  weights.py --kair-init for swinir_cls / swinir_real into a scratch
             models/ folder, then KAIR's OWN ModelBase.load_network (the call
             model_plain/model_gan.load() makes, strict, param_key 'params') puts
             0_G.pth and 0_E.pth into KAIR's network_swinir.SwinIR built with the
             same arguments. This is also the check that KAIR's copy of
             network_swinir.py (which differs from the official one; gate env
             swinir_src) has the same parameters for these two configurations.
  vgg19      the perceptual-loss file into torchvision's vgg19(), strict.

Not here: that write_arm.py's predict() reproduces the official inference
scripts to 1 LSB. That is step 4, where predict() is written.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.util import paths  # noqa: E402
from variants.sr_finetune import nets  # noqa: E402
from variants.sr_finetune.gates import probe as child  # noqa: E402

GATE_DIR = os.path.join(paths.REPO, "runs", "sr_ft", "gate")
SCRATCH = os.path.join(GATE_DIR, "weights_scratch")
WEIGHTS_PY = os.path.join(paths.VARIANTS, "sr_finetune", "weights.py")
USE = {"basicsr": paths.use_basicsr, "swinir": paths.use_swinir_official, "kair": paths.use_kair}


def strict_load(net, state):
    missing, unexpected = net.load_state_dict(state, strict=False)
    return {"missing": list(missing)[:10], "unexpected": list(unexpected)[:10],
            "ok": not missing and not unexpected}


def forward(net):
    import torch
    with torch.no_grad():
        y = net.eval()(torch.rand(1, 3, 32, 32))
    return {"out_shape": list(y.shape), "finite": bool(torch.isfinite(y).all()),
            "ok": list(y.shape) == [1, 3, 128, 128] and bool(torch.isfinite(y).all())}


def probe(code):
    """Child: one code base on sys.path; every weight whose arch comes from it."""
    USE[code]()
    out = {}
    if code == "kair":
        from models.model_base import ModelBase
        for arm in ("swinir_cls", "swinir_real"):
            arch = nets.WEIGHTS[arm]["arch"]
            models_dir = os.path.join(SCRATCH, arm, "models")
            base = ModelBase({"path": {"models": models_dir}, "gpu_ids": None, "is_train": False})
            res = {}
            for n in ("G", "E"):
                net = nets.build(arch)          # KAIR's models.network_swinir here
                try:
                    base.load_network(os.path.join(models_dir, "0_%s.pth" % n), net, strict=True)
                    res[n] = {"ok": True}
                except Exception as e:  # noqa: BLE001
                    res[n] = {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:300])}
            import models.network_swinir as kair_net
            res["network_file"] = os.path.relpath(kair_net.__file__, paths.REPO)
            res["ok"] = res["G"]["ok"] and res["E"]["ok"]
            out[arm] = res
    else:
        for name, w in nets.WEIGHTS.items():
            if not w["arch"] or nets.ARCHS[w["arch"]]["code"] != code:
                continue
            net = nets.build(w["arch"])
            res = strict_load(net, nets.load_state(w["file"], w["key"]))
            if res["ok"]:
                res["forward"] = forward(net)
                res["ok"] = res["forward"]["ok"]
            mod = sys.modules[nets.ARCHS[w["arch"]]["module"]]
            res["network_file"] = os.path.relpath(mod.__file__, paths.REPO)
            res["parameters"] = int(sum(p.numel() for p in net.parameters()))
            out[name] = res
    child.emit({"ok": all(r["ok"] for r in out.values()), "weights": out})


def vgg19():
    import torchvision
    w = nets.WEIGHTS["vgg19"]
    res = strict_load(torchvision.models.vgg19(), nets.load_state(w["file"], None))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", choices=sorted(USE), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.probe:
        return probe(args.probe)

    checks = {}
    r = subprocess.run([sys.executable, WEIGHTS_PY, "--check"])
    with open(os.path.join(nets.WEIGHTS_DIR, "manifest.json")) as f:
        man = json.load(f)
    checks["manifest"] = {"ok": r.returncode == 0 and not man["problems"],
                          "problems": man["problems"],
                          "sha256": {k: v.get("sha256") for k, v in man["weights"].items()}}

    checks["strict"] = {}
    for code in ("basicsr", "swinir"):
        checks["strict"][code] = child.run(os.path.abspath(__file__), code)
    checks["strict"]["ok"] = all(v["ok"] for v in checks["strict"].values())

    shutil.rmtree(SCRATCH, ignore_errors=True)
    inits = {}
    for arm in ("swinir_cls", "swinir_real"):
        d = os.path.join(SCRATCH, arm, "models")
        inits[arm] = subprocess.run([sys.executable, WEIGHTS_PY, "--kair-init", arm,
                                     "--models-dir", d]).returncode == 0
    k = child.run(os.path.abspath(__file__), "kair")
    checks["kair_init"] = {"ok": all(inits.values()) and k["ok"], "written": inits, "load": k}
    shutil.rmtree(SCRATCH, ignore_errors=True)

    checks["vgg19"] = vgg19()

    report = {"checks": checks, "failed": [n for n, c in checks.items() if not c["ok"]]}
    report["pass"] = not report["failed"]
    os.makedirs(GATE_DIR, exist_ok=True)
    path = os.path.join(GATE_DIR, "weights.json")
    with open(path + ".partial", "w") as f:
        json.dump(report, f, indent=2)
    os.replace(path + ".partial", path)
    for n, c in checks.items():
        print("  %-10s %s" % (n, "ok" if c["ok"] else "FAIL"))
    print("[gate weights] %s -> %s" % ("PASS" if report["pass"] else "FAIL", path))
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
