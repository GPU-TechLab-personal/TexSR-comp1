#!/usr/bin/env python3
"""
Gate for experiments 2/3: every arm config trains end to end for a few
iterations on a scratch training set - both stages (PSNR warm-up, then GAN),
quick eval, checkpoint, then RESUME from that checkpoint and finish.

    python experiments/data/build_train_set.py --limit 16 --out /expdata/_gate_train   # outside variants/
    python variants/experiments/gates/gate_train_smoke.py

The scratch set is $EXPDATA/_gate_train (default /expdata/_gate_train, as
before); with EXPDATA pointed elsewhere the runs land there too, so a test can
keep every write out of /expdata/runs (variants/experiments/README.md).

Reports peak memory and it/s per arm (the batch the real runs will use), and
the pretrained-init report (tensors loaded / heads re-initialised).
"""
import glob
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.experiments.common import paths   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.normpath(os.path.join(_HERE, os.pardir, "esrgan_map_degradation", "core", "train.py"))
CONFIGS = sorted(glob.glob(os.path.join(_HERE, os.pardir, "esrgan_map_degradation", "configs", "*.yaml"))
                 + glob.glob(os.path.join(_HERE, os.pardir, "esrgan_render_loss", "configs", "*.yaml")))
SET = ["data.root=%s" % os.path.join(paths.EXPDATA, "_gate_train"), "data.workers=2", "train.psnr_iters=10",
       "log.every=10", "eval.every=20", "save.every=20", "train.milestones=[15]"]


def main():
    fails = []
    only = sys.argv[1:]                     # optional substrings, e.g. "3-1 3-2"
    for cfg in [c for c in CONFIGS if not only or any(o in c for o in only)]:
        name = "_smoke_" + os.path.basename(cfg).replace(".yaml", "")
        exp = "exp3" if "esrgan_render_loss" in cfg else "exp2"
        run = os.path.join(paths.RUNS, exp, name)
        shutil.rmtree(run, ignore_errors=True)
        ok = True
        for niter in (20, 30):                     # 20 then resume to 30
            p = subprocess.run([sys.executable, TRAIN, "--config", cfg,
                                "--run-name", name, "--set", *SET, "train.niter=%d" % niter],
                               capture_output=True, text=True)
            if p.returncode != 0:
                ok = False
                print("FAIL %s (niter %d)\n%s" % (name, niter, p.stderr[-1500:]))
                break
            if niter == 30 and "resumed at iter 20" not in p.stdout:
                ok = False
                print("FAIL %s: did not resume\n%s" % (name, p.stdout[-800:]))
        if not ok:
            fails.append(name)
            continue
        logs = [json.loads(l) for l in open(os.path.join(run, "log.jsonl"))]
        ev = [json.loads(l) for l in open(os.path.join(run, "eval.jsonl"))]
        setup = json.load(open(os.path.join(run, "setup.json")))
        gan = [l for l in logs if l["stage"] == "gan"]
        init = setup["init"]
        print("PASS %-26s stages %s  %.2f it/s  peak %5.0f MB  evals %d  init %s%s" % (
            name, sorted({l["stage"] for l in logs}), gan[-1]["it_per_s"], max(l["peak_mem_mb"] for l in logs),
            len(ev), init["mode"],
            "" if init["mode"] == "scratch" else " (%d tensors, re-init %s)" % (init["loaded"], init["skipped"])))
        print("     eval " + "  ".join("%s %.4f" % (k, v) for k, v in ev[-1].items() if k not in ("iter",)))
    print("\n%s" % ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
