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
  * the calibration's gradient norms within --calib-rtol (cuDNN's backward is
    not bit-deterministic run to run - old vs old differs by as much), and
    w_star itself exactly; w_star_unrounded is also checked against the
    committed runs/deeptex/render_gate/calibration.json (9.541...).

Fields that name a time, a path or the script are skipped (SKIP_KEYS).
GPU: one job at a time - it refuses to start while another texsr container runs.
"""

import argparse
import datetime
import json
import math
import os
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
             "elapsed_s", "time", "timestamp", "host", "tf_version", "torch_version", "path"}
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


def walk(a, b, path, out, rtol):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP_KEYS:
                continue
            if k not in a or k not in b:
                out.append((path + "/" + k, "missing on one side", None))
                continue
            walk(a[k], b[k], path + "/" + k, out, rtol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "length %d vs %d" % (len(a), len(b)), None))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, "%s[%d]" % (path, i), out, rtol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        if a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)):
            return
        rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
        if rel > rtol:
            out.append((path, "%r vs %r" % (a, b), rel))
    elif a != b:
        out.append((path, "%r vs %r" % (str(a)[:80], str(b)[:80]), None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="parity,calibrate")
    ap.add_argument("--calib-rtol", type=float, default=2e-3)
    args = ap.parse_args()
    busy = subprocess.run(["docker", "ps", "--format", "{{.Image}}"], stdout=subprocess.PIPE
                          ).stdout.decode().split()
    if any(i.startswith("texsr/") for i in busy):
        sys.exit("L4: another texsr container is running (%s) - one GPU job at a time" % busy)
    what = args.what.split(",")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = W + "/runs/variants_verify/l4/" + stamp
    logs = host(base)
    os.makedirs(logs, exist_ok=True)
    codes = {}
    for side in ("old", "new"):
        os.makedirs(host(base + "/" + side), exist_ok=True)
        codes[side] = run(side, base + "/" + side, logs, what)
        print("[l4] %s exit codes %s" % (side, codes[side]), flush=True)

    rows, n_fail = [], 0
    if codes["old"] != codes["new"]:
        rows.append(("exit codes", "FAIL", "%s vs %s" % (codes["old"], codes["new"])))
        n_fail += 1
    names = sorted(f for f in os.listdir(host(base + "/old")) if f.endswith(".json"))
    for n in names:
        a = json.load(open(os.path.join(host(base + "/old"), n)))
        pb = os.path.join(host(base + "/new"), n)
        if not os.path.isfile(pb):
            rows.append((n, "FAIL", "new side wrote no %s" % n)); n_fail += 1; continue
        b = json.load(open(pb))
        diffs = []
        walk(a, b, "", diffs, args.calib_rtol if n == "calibration.json" else 0.0)
        if n == "calibration.json":
            ref = json.load(open(os.path.join(REPO, "runs", "deeptex", "render_gate", n)))
            for side, c in (("old", a), ("new", b)):
                if c.get("w_star") != ref.get("w_star"):
                    diffs.append(("/w_star", "%s %r vs committed %r" % (side, c.get("w_star"),
                                                                        ref.get("w_star")), None))
            detail = "w_star %s / %s, unrounded %.6g / %.6g (committed %.6g)" % (
                a.get("w_star"), b.get("w_star"), a.get("w_star_unrounded") or float("nan"),
                b.get("w_star_unrounded") or float("nan"), ref.get("w_star_unrounded"))
        else:
            detail = "exact"
        v = "PASS" if not diffs else "FAIL"
        n_fail += v == "FAIL"
        rows.append((n, v, detail + "".join("<br>`%s` %s" % (p, m) for p, m, _ in diffs[:8])))
    md = ["# L4 render gates, old vs new (%s)" % stamp, "", "| file | verdict | detail |",
          "|---|---|---|"] + ["| %s | %s | %s |" % r for r in rows]
    md += ["", "exit codes: old %s, new %s" % (codes["old"], codes["new"]),
           "", "%d files, %d FAIL" % (len(names), n_fail)]
    with open(os.path.join(logs, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    for r in rows:
        print("%s %s  %s" % (r[1], r[0], r[2].replace("<br>", "\n       ")))
    print("L4: %d FAIL -> %s" % (n_fail, os.path.join(logs, "report.md")))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
