#!/usr/bin/env python3
"""L4: the render gates, old against new (GPU).

Host-side driver. Runs, with the argv run_deeptex.ps1 -Stage gate-render uses,

  parity  render_gate --part export -> render_parity_torch --part all ->
          render_gate --part grads,flow,grid,chunks,mask,graph
  calibrate  render_calibrate on the six trained states (matsynth_all and
          matsynth_chratio at 30k/35k/40k), 20 batches, seed 0

once with scripts/ and once with variants/, each into its own folder under
runs/variants_verify/l4/<stamp>/{old,new}/ (never runs/deeptex/render_gate).
Then compares every JSON the two wrote, number by number:

  * exact for the parity / gate JSONs (same ops on the same inputs);
  * the calibration: cuDNN's backward is not bit-deterministic run to run, so
    the OLD calibration runs twice, and the median / p90 / max relative
    difference over all numbers new-vs-old may each be at most
    max(--calib-rtol, 10 x the same statistic old-vs-old) - the aggregate rule
    L1 uses for the render cases (measured 2026-10-31: old vs old differs in
    all 1,765 numbers, unrounded w 9.553 / 9.565). w_star itself exactly, both against each other and against
    the committed runs/deeptex/render_gate/calibration.json (w_star 10).
  * parity_tf.npz array by array (all must be equal) and its manifest with
    "created" removed: the file's sha256 changes on every export because the
    manifest stamps the time, so export.json's two sha256 fields are skipped.
  * GPU memory peaks (chunks.json "*peak*_mb") are measurements of the
    allocator, not results: reported, not compared.

Fields that name a time, a path or the script are skipped (SKIP_KEYS).
--reuse <stamp> compares an earlier run's old/new folders again, running only
what is missing (e.g. the second old calibration).
GPU: one job at a time - it refuses to start while another texsr container runs.
"""

import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir))
W = "/workspace"
EVAL_ROOT = W + "/runs/matsynth"
MS_PAIRS = W + "/runs/deeptex/datasets/matsynth"
GROUPED = W + "/runs/deeptex/datasets/matsynth_grouped/matsynth_grouped_lrx4.tfrecord"
R1_OLD = W + "/configs/deeptex/esrgan_matsynth_render_r1.yaml"
R1_NEW = W + "/variants/deeptex/render_loss/r1.yaml"
SIDES = {
    "old": {"gate": W + "/scripts/deeptex/render_gate.py",
            "torch": W + "/scripts/matup/render_parity_torch.py",
            "calib": W + "/scripts/deeptex/render_calibrate.py", "cfg": R1_OLD},
    "new": {"gate": W + "/variants/deeptex/render_loss/gates/render_gate.py",
            "torch": W + "/variants/deeptex/render_loss/gates/render_parity_torch.py",
            "calib": W + "/variants/deeptex/render_loss/gates/render_calibrate.py", "cfg": R1_NEW},
}
IMAGES = {"deeptex": ("texsr/deeptex:tf2.10.1", W + "/external/DeepTextures"),
          "matup": ("texsr/matup:torch2.0.0-cu117", W + "/external/matup_egsr")}
SKIP_KEYS = {"created", "seconds", "script", "cfg", "out", "npz", "tfrecord", "prefix",
             "elapsed_s", "time", "timestamp", "host", "tf_version", "torch_version", "path",
             "npz_sha256", "sha256"}
MEASURED = re.compile(r"peak.*_mb$")
STEPS = (30000, 35000, 40000)


def host(ctr):
    return os.path.join(REPO, *ctr[len(W) + 1:].split("/"))


def docker(image, cmd, log):
    img, wd = IMAGES[image]
    full = ["docker", "run", "--rm", "--gpus", "all", "--shm-size", "8g",
            "-v", "%s:%s" % (REPO, W), "-w", wd, img] + cmd
    p = subprocess.run(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       env=dict(os.environ, MSYS_NO_PATHCONV="1"))
    with open(log, "w", encoding="utf-8") as f:
        f.write(p.stdout.decode("utf-8", "replace"))
    return p.returncode


def ckpt_prefixes():
    """The six calibration states, by the step tensor in each checkpoint's name:
    the matsynth arms keep every save in checkpoints_all/ (ckpt-N, N = step/5000)."""
    out = []
    for arm in ("matsynth_all", "matsynth_chratio"):
        for s in STEPS:
            n = s // 5000
            for sub in ("checkpoints_all", "checkpoints"):
                p = "%s/runs/deeptex/%s/%s/esrgan_40k/ckpt-%d" % (W, arm, sub, n)
                if os.path.isfile(host(p) + ".index"):
                    out += ["--ckpt", p]
                    break
            else:
                sys.exit("L4: no %s checkpoint at step %d" % (arm, s))
    return out


def run(side, out, logs, what):
    s = SIDES[side]
    codes = {}
    if "parity" in what:
        common = ["--cfg", s["cfg"], "--eval-root", EVAL_ROOT, "--pairs-root", MS_PAIRS, "--out", out]
        codes["export"] = docker("deeptex", ["python", s["gate"], "--part", "export"] + common,
                                 os.path.join(logs, side + "-export.log"))
        codes["torch"] = docker("matup", ["python", s["torch"], "--part", "all", "--npz",
                                          out + "/parity_tf.npz", "--eval-root", EVAL_ROOT,
                                          "--out", out], os.path.join(logs, side + "-torch.log"))
        codes["tf"] = docker("deeptex", ["python", s["gate"], "--part",
                                         "grads,flow,grid,chunks,mask,graph"] + common,
                             os.path.join(logs, side + "-tf.log"))
    if "calibrate" in what:
        codes["calibrate"] = docker("deeptex", ["python", s["calib"], "--cfg", s["cfg"]]
                                    + ckpt_prefixes() + ["--batches", "20", "--seed", "0",
                                                         "--tfrecord", GROUPED,
                                                         "--out", out + "/calibration.json"],
                                    os.path.join(logs, side + "-calibrate.log"))
    return codes


def walk(a, b, path, out, rtol, measured=None):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP_KEYS:
                continue
            if measured is not None and MEASURED.search(k):
                measured.append("%s/%s %r / %r" % (path, k, a.get(k), b.get(k)))
                continue
            if k not in a or k not in b:
                out.append((path + "/" + k, "missing on one side", None))
                continue
            walk(a[k], b[k], path + "/" + k, out, rtol, measured)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "length %d vs %d" % (len(a), len(b)), None))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, "%s[%d]" % (path, i), out, rtol, measured)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        if a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)):
            return
        rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
        if rel > rtol:
            out.append((path, "%r vs %r" % (a, b), rel))
    elif a != b:
        out.append((path, "%r vs %r" % (str(a)[:80], str(b)[:80]), None))


def npz_diffs(pa, pb):
    import numpy as np
    a, b = np.load(pa, allow_pickle=False), np.load(pb, allow_pickle=False)
    out = []
    if sorted(a.files) != sorted(b.files):
        return [("/files", "%s vs %s" % (sorted(a.files), sorted(b.files)), None)], len(a.files)
    for k in sorted(a.files):
        x, y = a[k], b[k]
        if k == "manifest":
            mx, my = json.loads(str(x)), json.loads(str(y))
            mx.pop("created", None)
            my.pop("created", None)
            walk(mx, my, "/manifest", out, 0.0)
        elif x.dtype != y.dtype or x.shape != y.shape or \
                not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            out.append(("/" + k, "differs", None))
    return out, len(a.files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="parity,calibrate")
    ap.add_argument("--calib-rtol", type=float, default=2e-3)
    ap.add_argument("--reuse", default="", help="stamp of an earlier run to compare again")
    args = ap.parse_args()
    busy = subprocess.run(["docker", "ps", "--format", "{{.Image}}"], stdout=subprocess.PIPE
                          ).stdout.decode().split()
    if any(i.startswith("texsr/") for i in busy):
        sys.exit("L4: another texsr container is running (%s) - one GPU job at a time" % busy)
    what = args.what.split(",")
    stamp = args.reuse or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = W + "/runs/variants_verify/l4/" + stamp
    logs = host(base)
    os.makedirs(logs, exist_ok=True)
    codes = {}
    sides = [("old", "old"), ("new", "new")]
    if "calibrate" in what:
        sides.append(("old2", "old"))
    for name, side in sides:
        d = base + "/" + name
        codes_file = os.path.join(logs, name + "-codes.json")
        if args.reuse and os.path.isfile(codes_file):
            codes[name] = json.load(open(codes_file))
            print("[l4] %s reused %s" % (name, codes[name]), flush=True)
            continue
        if args.reuse and name in ("old", "new") and os.path.isdir(host(d)):
            # a run from before the codes file existed: its report gave every code
            codes[name] = {k: 0 for k in (["export", "torch", "tf"] if "parity" in what else [])
                           + (["calibrate"] if "calibrate" in what else [])}
            print("[l4] %s reused (codes from the earlier report) %s" % (name, codes[name]), flush=True)
            continue
        os.makedirs(host(d), exist_ok=True)
        codes[name] = run(side, d, logs, what if name != "old2" else ["calibrate"])
        json.dump(codes[name], open(codes_file, "w"))
        print("[l4] %s exit codes %s" % (name, codes[name]), flush=True)

    rows, n_fail = [], 0
    if codes["old"] != codes["new"] or \
            ("old2" in codes and codes["old2"].get("calibrate") != codes["old"].get("calibrate")):
        rows.append(("exit codes", "FAIL", "%s vs %s" % (codes["old"], codes["new"])))
        n_fail += 1
    names = sorted(f for f in os.listdir(host(base + "/old")) if f.endswith(".json"))
    for n in names:
        a = json.load(open(os.path.join(host(base + "/old"), n)))
        pb = os.path.join(host(base + "/new"), n)
        if not os.path.isfile(pb):
            rows.append((n, "FAIL", "new side wrote no %s" % n)); n_fail += 1; continue
        b = json.load(open(pb))
        diffs, measured = [], []
        if n == "calibration.json":
            raw, oo = [], []
            walk(a, b, "", raw, 0.0)
            a2 = json.load(open(os.path.join(host(base + "/old2"), n)))
            walk(a, a2, "", oo, 0.0)
            # Aggregate, as L1 does: one old-vs-old pair is too few samples to give
            # each of ~1,800 numbers its own spread (a field that happened to land
            # close in that pair would get a tolerance near zero).
            diffs += [(p, m, r) for p, m, r in raw + oo if r is None]
            rel_no = sorted(r for _, _, r in raw if r is not None)
            rel_oo = sorted(r for _, _, r in oo if r is not None)

            def q(v, f):
                return v[min(len(v) - 1, int(f * len(v)))] if v else 0.0
            stats = []
            for name, f in (("median", 0.5), ("p90", 0.9), ("max", 1.0)):
                no, oo_ = q(rel_no, f), q(rel_oo, f)
                tol = max(args.calib_rtol, 10 * oo_)
                stats.append("%s %.3g vs %.3g" % (name, no, oo_))
                if no > tol:
                    diffs.append(("/%s rel diff" % name, "new-vs-old %.3g > max(rtol, 10 x old-vs-old %.3g)"
                                  % (no, oo_), no))
        else:
            walk(a, b, "", diffs, 0.0, measured)
        if n == "calibration.json":
            ref = json.load(open(os.path.join(REPO, "runs", "deeptex", "render_gate", n)))
            for side, c in (("old", a), ("new", b)):
                if c.get("w_star") != ref.get("w_star"):
                    diffs.append(("/w_star", "%s %r vs committed %r" % (side, c.get("w_star"),
                                                                        ref.get("w_star")), None))
            detail = ("w_star %s / %s, unrounded old %.6g, old again %.6g, new %.6g (committed %.6g); "
                      "%d numbers differ old-vs-old, %d new-vs-old; relative difference new-vs-old "
                      "vs old-vs-old: %s") % (
                a.get("w_star"), b.get("w_star"), a.get("w_star_unrounded") or float("nan"),
                a2.get("w_star_unrounded") or float("nan"),
                b.get("w_star_unrounded") or float("nan"), ref.get("w_star_unrounded"),
                len(oo), len(raw), ", ".join(stats))
        else:
            detail = "exact" + ("; measured, not compared: " + ", ".join(measured) if measured else "")
        v = "PASS" if not diffs else "FAIL"
        n_fail += v == "FAIL"
        rows.append((n, v, detail + "".join("<br>`%s` %s" % (p, m) for p, m, _ in diffs[:8])))
    pa, pb = host(base + "/old/parity_tf.npz"), host(base + "/new/parity_tf.npz")
    if os.path.isfile(pa):
        if not os.path.isfile(pb):
            rows.append(("parity_tf.npz", "FAIL", "new side wrote none"))
            n_fail += 1
        else:
            d, nfiles = npz_diffs(pa, pb)
            rows.append(("parity_tf.npz", "FAIL" if d else "PASS",
                         "%d arrays equal, manifest equal without 'created'" % nfiles if not d else
                         "".join("<br>`%s` %s" % (p, m) for p, m, _ in d[:8])))
            n_fail += bool(d)
    md = ["# L4 render gates, old vs new (%s)" % stamp, "", "| file | verdict | detail |",
          "|---|---|---|"] + ["| %s | %s | %s |" % r for r in rows]
    md += ["", "exit codes: %s" % codes,
           "", "%d files, %d FAIL" % (len(names), n_fail)]
    with open(os.path.join(logs, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    for r in rows:
        print("%s %s  %s" % (r[1], r[0], r[2].replace("<br>", "\n       ")))
    print("L4: %d FAIL -> %s" % (n_fail, os.path.join(logs, "report.md")))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
