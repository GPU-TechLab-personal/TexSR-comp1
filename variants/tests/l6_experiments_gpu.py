#!/usr/bin/env python3
"""L6 (experiments), GPU layers. Short: minutes, not hours.

    variants/tests/run_experiments_l6.sh gpu python variants/tests/l6_experiments_gpu.py --out runs/variants_verify/experiments/<stamp>

  matup   matup_driver.optimise, old vs new, on one eval12 material (SwinIR cache,
          read-only), seed 1996, --epochs epochs (1 = 1922 Adam steps): the
          plain MatUp loop and the exp4 hook (grf_stats variant B, lambda 0.003).
          Judged with allclose (the one place equality is not asked for: the
          8-worker DataLoader and CUDA's atomics make even old vs old differ),
          and reported with the measured old-vs-old floor next to it.
  smoke   gate_train_smoke.py on the NEW path (8 arms x 20 iters + resume to 30,
          both stages, quick eval, checkpoints), with EXPDATA=/vvx so every write
          lands in variants_verify.
  compat  an OLD-trainer checkpoint is resumed by the NEW trainer (2-0, 2-2,
          3-2: the three D layouts shared / ds.<map> / joint) and evaluated by
          the old and the new evaluate.py: the two results.json must agree.
Writes <out>/gpu.json. Exit 0 = every check PASS.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

VVX = "/vvx"
ATOL = 1e-3                     # on MatUp's internal maps: albedo_lin, metallic, normal, alpha in [0,1] / unit
MATERIAL = "acg_wood_floor_060"
SMOKE_SET = ["data.root=%s/_gate_train" % VVX, "data.workers=2", "train.psnr_iters=10",
             "log.every=10", "eval.every=20", "save.every=20", "train.milestones=[15]"]
RESULTS = []


def record(layer, name, ok, detail="", **extra):
    RESULTS.append(dict({"layer": layer, "check": name, "pass": bool(ok), "detail": detail}, **extra))
    print("  %s %-7s %-56s %s" % ("PASS" if ok else "FAIL", layer, name, detail), flush=True)


# ----------------------------------------------------------------------- matup

def run_driver(D, factory, epochs, dev):
    from variants.experiments.common import paths
    root = os.path.join(paths.SR_CACHE, "eval12", MATERIAL)
    mat = os.path.join(paths.MATERIALS, "eval12", MATERIAL)
    args = D.matup_args(os.path.join(mat, "LR"), os.path.join(root, "radiance_LR_L100"),
                        os.path.join(root, "radiance_swinir_L100"), "/vv/unused", nb_epochs=epochs)
    t0 = time.perf_counter()
    maps, model, ds, hist = D.optimise(args, dev, factory, seed=1996, log=lambda *a: None)
    out = [m.detach().float().cpu() for m in maps[:4]]
    del model, ds
    torch.cuda.empty_cache()
    return out, hist, time.perf_counter() - t0


def maxdiff(a, b):
    return [float((x - y).abs().max()) for x, y in zip(a, b)]


def part_matup(epochs):
    import experiments.common.matup_driver as OD
    import experiments.exp4_grf_stats.grf_stats as OG
    from variants.experiments import matup_driver as ND
    from variants.experiments.exp4_grf_stats import grf_stats as NG
    dev = torch.device("cuda:0")
    names = ["albedo_lin", "metallic", "normal", "alpha"]
    for label, fo, fn in (("orig", lambda: None, lambda: None),
                          ("B_l0.003 (extra_loss hook)", lambda: OG.factory("B", 0.003), lambda: NG.factory("B", 0.003))):
        old, ho, to = run_driver(OD, fo(), epochs, dev)
        new, hn, tn = run_driver(ND, fn(), epochs, dev)
        again, ha, ta = run_driver(OD, fo(), epochs, dev)
        d_new, d_floor = maxdiff(old, new), maxdiff(old, again)
        ok = all(torch.allclose(a, b, rtol=0, atol=ATOL) for a, b in zip(old, new))
        lo = [h["loss_matup"] for h in ho]
        ln = [h["loss_matup"] for h in hn]
        detail = ("max|new-old| %s; old-vs-old floor %s; loss_matup old %s new %s; %.0f/%.0f/%.0fs"
                  % (" ".join("%s %.2e" % kv for kv in zip(names, d_new)),
                     " ".join("%.2e" % v for v in d_floor),
                     ["%.6f" % v for v in lo], ["%.6f" % v for v in ln], to, tn, ta))
        record("matup", "optimise %s, %d epoch(s), seed 1996, allclose atol %g" % (label, epochs, ATOL),
               ok, detail, max_abs_new_vs_old=dict(zip(names, d_new)),
               max_abs_old_vs_old=dict(zip(names, d_floor)), atol=ATOL,
               loss_matup_old=lo, loss_matup_new=ln,
               loss_extra_old=[h["loss_extra"] for h in ho], loss_extra_new=[h["loss_extra"] for h in hn])


# ------------------------------------------------------------- smoke / compat

def scratch_train_set(n=4):
    """A complete-looking training root: n built materials + materials.json."""
    dst = os.path.join(VVX, "_gate_train")
    names = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob("/expdata/train/*/HR"))[:n]
    for nm in names:
        if not os.path.isdir(os.path.join(dst, nm, "HR")):
            shutil.copytree(os.path.join("/expdata/train", nm), os.path.join(dst, nm))
    with open(os.path.join(dst, "materials.json"), "w") as fh:
        json.dump({"materials": names, "note": "variants L6 scratch set"}, fh)
    return names


def env():
    return dict(os.environ, EXPDATA=VVX)


def part_smoke():
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, os.path.join(REPO, "variants", "experiments", "gates", "gate_train_smoke.py")],
                       capture_output=True, text=True, env=env(), cwd=REPO)
    print(p.stdout[-6000:], p.stderr[-3000:] if p.returncode else "", flush=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith(("PASS ", "FAIL "))]
    record("smoke", "gate_train_smoke.py (new path, 8 arms, resume)", p.returncode == 0 and "ALL PASS" in p.stdout,
           "exit %d, %d arm lines, %.0fs" % (p.returncode, len(lines), time.perf_counter() - t0), arms=lines)


def train(script, cfg, name, niter):
    p = subprocess.run([sys.executable, script, "--config", cfg, "--run-name", name, "--set", *SMOKE_SET,
                        "train.niter=%d" % niter], capture_output=True, text=True, env=env(), cwd=REPO)
    return p


def evaluate(script, run):
    return subprocess.run([sys.executable, script, "--run", run], capture_output=True, text=True, env=env(), cwd=REPO)


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, p + "/" + str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flat(v, p + "/%d" % i))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[p] = float(d)
    return out


def part_compat():
    old_train = os.path.join(REPO, "experiments", "exp2_esrgan", "train.py")
    new_train = os.path.join(REPO, "variants", "experiments", "exp2_esrgan", "core", "train.py")
    old_eval = os.path.join(REPO, "experiments", "exp2_esrgan", "evaluate.py")
    new_eval = os.path.join(REPO, "variants", "experiments", "exp2_esrgan", "core", "evaluate.py")
    for cfg, exp in (("exp2_esrgan/configs/2-0_base_scratch.yaml", "exp2"),
                     ("exp2_esrgan/configs/2-2_separate.yaml", "exp2"),
                     ("exp3_render_loss/configs/3-2_render_jointD.yaml", "exp3")):
        arm = os.path.basename(cfg).replace(".yaml", "")
        name = "_compat_" + arm
        run = os.path.join(VVX, "runs", exp, name)
        shutil.rmtree(run, ignore_errors=True)
        t0 = time.perf_counter()
        p1 = train(old_train, os.path.join(REPO, "experiments", cfg), name, 20)
        p2 = train(new_train, os.path.join(REPO, "variants", "experiments", cfg), name, 30)
        ok = p1.returncode == 0 and p2.returncode == 0 and "resumed at iter 20 from" in p2.stdout
        record("compat", "%s: old ckpt iter 20 -> new trainer resumes to 30" % arm, ok,
               "old exit %d, new exit %d, %s (%.0fs)" % (p1.returncode, p2.returncode,
                                                       "resumed" if "resumed at iter 20" in p2.stdout else "NOT resumed",
                                                       time.perf_counter() - t0))
        if not ok:
            print(p1.stderr[-2000:], p2.stdout[-1500:], p2.stderr[-2000:])
            continue
        res = os.path.join(run, "eval", "iter_30", "results.json")
        t0 = time.perf_counter()
        e1 = evaluate(old_eval, "%s/%s" % (exp, name))
        r_old = json.load(open(res)) if e1.returncode == 0 else None
        e2 = evaluate(new_eval, "%s/%s" % (exp, name))
        r_new = json.load(open(res)) if e2.returncode == 0 else None
        if r_old is None or r_new is None:
            record("compat", "%s: evaluate.py old vs new on the iter-30 ckpt" % arm, False,
                   "old exit %d, new exit %d" % (e1.returncode, e2.returncode))
            print(e1.stderr[-1500:], e2.stderr[-1500:])
            continue
        fo, fn = flat(r_old["materials"]), flat(r_new["materials"])
        same_keys = list(fo) == list(fn)
        dmax = max((abs(fo[k] - fn[k]) for k in fo if k in fn), default=0.0)
        record("compat", "%s: evaluate.py old vs new on the iter-30 ckpt" % arm, same_keys and dmax == 0.0,
               "%d numbers, keys %s, max|d| %.3g (%.0fs)" % (len(fn), "==" if same_keys else "!=", dmax,
                                                          time.perf_counter() - t0),
               max_abs=dmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="*", default=["matup", "smoke", "compat"])
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    t0 = time.perf_counter()
    if "smoke" in args.parts or "compat" in args.parts:
        print("scratch training set:", scratch_train_set())
    for part in args.parts:
        print("[%s]" % part, flush=True)
        if part == "matup":
            part_matup(args.epochs)
        else:
            globals()["part_" + part]()
    n_fail = sum(not r["pass"] for r in RESULTS)
    path = os.path.join(args.out, "gpu.json")
    prev = json.load(open(path)) if os.path.isfile(path) else {"results": []}
    keep = [r for r in prev["results"] if r["layer"] not in {x["layer"] for x in RESULTS}]
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "seconds": time.perf_counter() - t0, "results": keep + RESULTS}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("L6 gpu: %d checks, %d FAIL (%.0fs)" % (len(RESULTS), n_fail, out["seconds"]))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
