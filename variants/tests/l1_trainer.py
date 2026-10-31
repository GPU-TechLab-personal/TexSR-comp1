#!/usr/bin/env python3
"""L1: the new trainer (paper loop + hooks) against the old one, on CPU.

Host-side driver (any Python 3.8+, needs docker). Every case runs twice in the
deeptex image, without a GPU, under tf.config.experimental.enable_op_determinism
(_det_run.py) and the same seed:

  old  python scripts/deeptex/deeptex_train.py ARGS
  new  python variants/deeptex/paper_loop/main.py --experiment X ARGS

with ARGS identical except the per-side output root. A case PASSES when
  * the exit codes are equal (and equal the expected one),
  * stdout is equal line for line after normalisation: the output root becomes
    <R>, and the wall-clock fields (steps_per_sec, eta_h, the disk-free GB, the
    progress bar's step/sec) become <t> - nothing else is masked,
  * the latest checkpoint holds the same variables with bit-identical values.

L1a is the same comparison for external/DeepTextures/train_esrgan.py (run by
runpy, unmodified) against main.py --experiment paper (loop.py + default hooks).

Fixtures come from l1_fixtures.py (run it first; run_all.ps1 does). Output:
runs/variants_verify/l1/<stamp>/{report.md, <case>/{old,new}/...}
"""

import argparse
import datetime
import os
import re
import subprocess
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir))
IMAGE = "texsr/deeptex:tf2.10.1"
W = "/workspace"
F = W + "/runs/variants_verify/l1/fixtures"
C = F + "/configs"
OLD = W + "/scripts/deeptex/deeptex_train.py"
NEW = W + "/variants/deeptex/paper_loop/main.py"
DET = W + "/variants/tests/_det_run.py"
UPSTREAM = W + "/external/DeepTextures/train_esrgan.py"

COMMON = ["--allow-cpu", "--seed", "7", "--log-every", "1",
          "--checkpoint-root", "{R}/ckpt", "--log-root", "{R}/logs"]
VAL = ["--val-hr", F + "/val/HR_val", "--val-lr", F + "/val/LRx4_val",
       "--val-every", "5", "--val-count", "2"]
RENDER = ["--stage", "esrgan", "--losses", "upstream", "--shuffle", "off",
          "--allow-dead-lr-schedule", "--no-val"]

# name, experiment, [argv per run], expected exit code, checkpoint subdir to compare
CASES = [
    ("fixed_psnr", "fixed",
     [["--cfg_path", C + "/psnr.yaml", "--stage", "psnr", "--max-steps", "5",
       "--save-steps", "5"] + COMMON + VAL], 0, "ckpt/psnr_texsr"),
    ("fixed_esrgan_corrected_shuffle", "fixed",
     [["--cfg_path", C + "/psnr.yaml", "--stage", "psnr", "--max-steps", "3",
       "--save-steps", "3", "--no-val"] + COMMON,
      ["--cfg_path", C + "/esrgan.yaml", "--stage", "esrgan", "--max-steps", "5",
       "--save-steps", "5", "--pretrain-dir", "{R}/ckpt/psnr_texsr"] + COMMON + VAL],
     0, "ckpt/esrgan_texsr"),
    ("fixed_esrgan_resume_3_5", "fixed",
     [["--cfg_path", C + "/esrgan.yaml", "--stage", "esrgan", "--max-steps", "3",
       "--save-steps", "3", "--pretrain-dir", "none", "--no-val"] + COMMON,
      ["--cfg_path", C + "/esrgan.yaml", "--stage", "esrgan", "--max-steps", "5",
       "--pretrain-dir", "none", "--no-val"] + COMMON], 0, "ckpt/esrgan_texsr"),
    ("asconfigured_upstream_noshuffle_deadlr", "fixed",
     [["--cfg_path", C + "/asconfigured.yaml", "--stage", "esrgan", "--losses", "upstream",
       "--shuffle", "off", "--allow-dead-lr-schedule", "--max-steps", "4",
       "--save-steps", "4", "--no-val"] + COMMON], 0, "ckpt/esrgan_asconfigured"),
    ("render_r0a_monitor", "render",
     [["--cfg_path", C + "/render_r0a.yaml", "--pretrain-dir", "none", "--w-render", "0",
       "--render-monitor-every", "2", "--max-steps", "4", "--save-steps", "4"]
      + RENDER + COMMON], 0, "ckpt/esrgan_40k"),
    # NONDET: the old trainer itself is not reproducible here (see NONDET below).
    ("render_r1", "render",
     [["--cfg_path", C + "/render_r1.yaml", "--pretrain-dir", "none",
       "--render-monitor-every", "2", "--max-steps", "4", "--save-steps", "4"]
      + RENDER + COMMON], 0, "ckpt/esrgan_40k"),
    ("render_r1_resume_3_5", "render",
     [["--cfg_path", C + "/render_r1.yaml", "--pretrain-dir", "none", "--max-steps", "3",
       "--save-steps", "3"] + RENDER + COMMON,
      ["--cfg_path", C + "/render_r1.yaml", "--pretrain-dir", "none", "--max-steps", "5",
       "--render-monitor-every", "100"] + RENDER + COMMON], 0, "ckpt/esrgan_40k"),
    ("render_overfit_from_pretrain", "render",
     [["--cfg_path", C + "/render_r0a.yaml", "--pretrain-dir", "none", "--w-render", "0",
       "--max-steps", "2", "--save-steps", "2", "--checkpoint-root", "{R}/pre"]
      + RENDER + COMMON[:-4] + ["--log-root", "{R}/logs"],
      ["--cfg_path", C + "/render_r1.yaml", "--pretrain-dir", "{R}/pre/esrgan_40k",
       "--overfit-batches", "1", "--render-monitor-every", "2", "--w-render", "9.54",
       "--max-steps", "4", "--save-steps", "4"] + RENDER + COMMON], 0, "ckpt/esrgan_40k"),
    # ---- failure paths: exit code and [fatal] text
    ("fail_pretrain_missing", "fixed",
     [["--cfg_path", C + "/esrgan.yaml", "--stage", "esrgan", "--max-steps", "2",
       "--pretrain-dir", "{R}/nothing_here", "--no-val"] + COMMON], 9, None),
    ("fail_dead_lr", "fixed",
     [["--cfg_path", C + "/asconfigured.yaml", "--stage", "esrgan", "--max-steps", "2",
       "--no-val"] + COMMON], 8, None),
    ("fail_render_flag_on_flat_config", "render",
     [["--cfg_path", C + "/esrgan.yaml", "--w-render", "1", "--max-steps", "2"]
      + RENDER + COMMON], 8, None),
    ("fail_no_tfrecord", "fixed",
     [["--cfg_path", C + "/psnr.yaml", "--stage", "psnr", "--tfrecord", "{R}/missing.tfrecord",
       "--max-steps", "2", "--no-val"] + COMMON], 5, None),
    ("fail_nonfinite_fixed", "fixed",
     [["--cfg_path", C + "/esrgan_nan.yaml", "--stage", "esrgan", "--max-steps", "3",
       "--pretrain-dir", "none", "--no-val"] + COMMON], 4, "ckpt/esrgan_texsr"),
    ("fail_nonfinite_render", "render",
     [["--cfg_path", C + "/render_r1_nan.yaml", "--pretrain-dir", "none", "--max-steps", "3"]
      + RENDER + COMMON], 4, "ckpt/esrgan_40k_nan_debug"),
]

# Cases on which the OLD trainer is not bit-reproducible against itself, under the
# same harness (op determinism, one thread, sequential maps, fixed seeds): from
# scratch with the render term on, the step on the third grouped material varies
# run to run (measured 2026-10-31: [terms] render 5.1079 / 5.1079 / 5.108, [render]
# step 4 l21 0.848005 / 0.847927 / 0.848328 over three old runs). The 2-step runs,
# r0a (w_render 0) and the overfit case (w_render 9.54, one cached batch) ARE
# bit-identical, old vs new. For these cases the old trainer runs twice and the
# verdict is: new-vs-old differs no more than NONDET_FACTOR x old-vs-old, in the
# checkpoint (max |d|) and in every number on the [metrics]/[terms]/[render]
# lines; every other line must still be identical.
NONDET = {"render_r1", "render_r1_resume_3_5"}
NONDET_FACTOR = 10.0

TIME = [(re.compile(r"(steps_per_sec|eta_h)=\S+"), r"\1=<t>"),
        (re.compile(r"[\d.]+ step/sec"), "<t> step/sec"),
        (re.compile(r"disk: [\d.]+ GB free"), "disk: <t> GB free")]


def host(ctr):
    return os.path.join(REPO, *ctr[len(W) + 1:].split("/"))


def docker(cmd, cwd=W + "/external/DeepTextures"):
    full = ["docker", "run", "--rm", "-v", "%s:%s" % (REPO, W), "-w", cwd,
            "-e", "TF_CPP_MIN_LOG_LEVEL=2", "-e", "PYTHONUNBUFFERED=1",
            "-e", "TF_ENABLE_ONEDNN_OPTS=0", "-e", "PYTHONHASHSEED=0", IMAGE] + cmd
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    p = subprocess.run(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def normalise(text, root):
    out = []
    for ln in re.split(r"[\r\n]+", text.replace(root, "<R>")):
        for rx, rep in TIME:
            ln = rx.sub(rep, ln)
        if ln.strip():
            out.append(ln.rstrip())
    return out


def run_side(script, pre, runs, root, log_dir):
    os.makedirs(host(root), exist_ok=True)
    code, text = 0, ""
    for i, argv in enumerate(runs):
        argv = [a.replace("{R}", root) for a in argv]
        code, out, err = docker(["python", DET, script] + pre + argv)
        with open(os.path.join(log_dir, "run%d.stdout" % i), "w", encoding="utf-8") as f:
            f.write(out)
        with open(os.path.join(log_dir, "run%d.stderr" % i), "w", encoding="utf-8") as f:
            f.write(err)
        text += "=== run %d exit %d\n" % (i, code) + out
        if code != 0:
            break
    return code, text


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        i = min(len(a), len(b))
        return i, a[i] if i < len(a) else "<end>", b[i] if i < len(b) else "<end>"
    return None


NUM = re.compile(r"(\w+)=(-?[\d.]+(?:e[-+]?\d+)?|nan|inf)")


def numbers(lines):
    out = []
    for ln in lines:
        if ln.startswith(("[metrics]", "[terms]", "[render]")):
            out.append((NUM.sub(r"\1=#", ln), [float(v) for _, v in NUM.findall(ln)]))
        else:
            out.append((ln, []))
    return out


def ckpt_maxabs(a_dir, b_dir):
    code, out, err = docker(["python", W + "/variants/tests/l1_ckpt_compare.py", a_dir, b_dir], cwd=W)
    last = [l for l in out.splitlines() if l.strip()]
    m = re.search(r"maxabs=(\S+)", last[0]) if last else None
    return code, (float(m.group(1)) if m else float("nan")), last, err


def compare(name, old, new, expect, ckpt, base):
    o_root, n_root = base + "/" + name + "/old", base + "/" + name + "/new"
    for r in (o_root, n_root):
        os.makedirs(host(r), exist_ok=True)
    oc, ot = old(o_root, host(o_root))
    nc, nt = new(n_root, host(n_root))
    problems = []
    if oc != nc or oc != expect:
        problems.append("exit codes old=%d new=%d expected=%d" % (oc, nc, expect))
    a, b = normalise(ot, o_root), normalise(nt, n_root)
    if name in NONDET:
        o2_root = base + "/" + name + "/old2"
        os.makedirs(host(o2_root), exist_ok=True)
        o2c, o2t = old(o2_root, host(o2_root))
        a2 = normalise(o2t, o2_root)
        na, nb, na2 = numbers(a), numbers(b), numbers(a2)
        # A non-numeric line may differ new-vs-old only where it also differs
        # old-vs-old: the resumed run's [init] hashes the (non-reproducible)
        # step-3 weights, so it is itself run-dependent.
        ta, tb, ta2 = [x for x, _ in na], [x for x, _ in nb], [x for x, _ in na2]
        if len(ta) != len(tb) or len(ta) != len(ta2):
            problems.append("stdout line counts differ: old %d, new %d, old2 %d"
                            % (len(ta), len(tb), len(ta2)))
        for i, (x, y, z) in enumerate(zip(ta, tb, ta2)):
            if x != y and x == z:
                problems.append("stdout line %d differs new-vs-old but not old-vs-old:"
                                "<br>old `%s`<br>new `%s`" % (i, x, y))
        spread_oo = spread_no = 0.0
        for (_, u), (_, v), (_, w) in zip(na, nb, na2):
            for x, y, z in zip(u, v, w):
                spread_oo = max(spread_oo, abs(x - z) / max(abs(x), 1e-12))
                spread_no = max(spread_no, abs(x - y) / max(abs(x), 1e-12))
        if spread_no > NONDET_FACTOR * max(spread_oo, 1e-12):
            problems.append("stdout numbers: new-vs-old rel %.3g > %g x old-vs-old %.3g"
                            % (spread_no, NONDET_FACTOR, spread_oo))
        ck = "logged numbers: old-vs-old rel %.3g, new-vs-old rel %.3g" % (spread_oo, spread_no)
        if ckpt:
            _, d_oo, _, _ = ckpt_maxabs(o_root + "/" + ckpt, o2_root + "/" + ckpt)
            _, d_no, last, _ = ckpt_maxabs(o_root + "/" + ckpt, n_root + "/" + ckpt)
            ck += "; ckpt max abs: old-vs-old %.3g, new-vs-old %.3g" % (d_oo, d_no)
            if not d_no <= NONDET_FACTOR * max(d_oo, 1e-12):
                problems.append("checkpoint: new-vs-old max |d| %.3g > %g x old-vs-old %.3g"
                                % (d_no, NONDET_FACTOR, d_oo))
        verdict = "PASS" if not problems else "FAIL"
        print("%s %-42s exit %d/%d, %d stdout lines (old not reproducible: %s)"
              % (verdict, name, oc, nc, len(a), ck), flush=True)
        for p in problems:
            print("       - " + p)
        return verdict, problems, len(a), ck
    d = first_diff(a, b)
    if d:
        problems.append("stdout line %d differs:<br>old `%s`<br>new `%s`" % d)
    if ckpt:
        code, _, last, err = ckpt_maxabs(o_root + "/" + ckpt, n_root + "/" + ckpt)
        if code != 0:
            problems.append("checkpoint: " + "<br>".join(last[:4] or err.splitlines()[-2:]))
        ck = last[0] if last else ""
    else:
        ck = "-"
    verdict = "PASS" if not problems else "FAIL"
    print("%s %-42s exit %d/%d, %d stdout lines" % (verdict, name, oc, nc, len(a)), flush=True)
    for p in problems:
        print("       - " + p.replace("<br>", "\n         "))
    return verdict, problems, len(a), ck


def gpu_case(base):
    """L1 GPU: the real r1 config (full RRDB, batch 2, real grouped record) for one
    step on the GPU, old vs new, without the determinism wrapper - cuDNN is not
    bit-deterministic, so: [init] identical, step-1 [terms] within 1e-6 relative."""
    busy = subprocess.run(["docker", "ps", "--format", "{{.Image}}"],
                          stdout=subprocess.PIPE).stdout.decode().split()
    if any(i.startswith("texsr/") for i in busy):
        sys.exit("L1 GPU: another texsr container is running (%s) - one GPU job at a time" % busy)
    outs = {}
    for side, script, pre in (("old", OLD, []), ("new", NEW, ["--experiment", "render"])):
        root = base + "/L1_gpu/" + side
        os.makedirs(host(root), exist_ok=True)
        argv = ["python", script] + pre + [
            "--cfg_path", W + "/configs/deeptex/esrgan_matsynth_render_r1.yaml", "--stage", "esrgan",
            "--losses", "upstream", "--shuffle", "off", "--allow-dead-lr-schedule",
            "--pretrain-dir", "none", "--no-val", "--log-every", "1", "--render-monitor-every", "1",
            "--max-steps", "1", "--save-steps", "1",
            "--checkpoint-root", root + "/ckpt", "--log-root", root + "/logs"]
        full = ["docker", "run", "--rm", "--gpus", "all", "--shm-size", "8g",
                "-v", "%s:%s" % (REPO, W), "-w", W + "/external/DeepTextures", IMAGE] + argv
        p = subprocess.run(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           env=dict(os.environ, MSYS_NO_PATHCONV="1"))
        out = p.stdout.decode("utf-8", "replace")
        with open(os.path.join(host(root), "run0.stdout"), "w", encoding="utf-8") as f:
            f.write(out)
        outs[side] = (p.returncode, out)
    problems = []
    (oc, ot), (nc, nt) = outs["old"], outs["new"]
    if oc or nc:
        problems.append("exit codes old=%d new=%d" % (oc, nc))
    pick = lambda t, pre: [l for l in t.splitlines() if l.startswith(pre)]
    if pick(ot, "[init]") != pick(nt, "[init]") or not pick(ot, "[init]"):
        problems.append("[init] differs: %s vs %s" % (pick(ot, "[init]"), pick(nt, "[init]")))
    worst = 0.0
    for pre in ("[terms]", "[render]"):
        a, b = pick(ot, pre), pick(nt, pre)
        if len(a) != len(b) or not a:
            problems.append("%s lines: %d vs %d" % (pre, len(a), len(b)))
            continue
        for x, y in zip(a, b):
            kx = re.findall(r"(\w+)=(\S+)", x)
            ky = re.findall(r"(\w+)=(\S+)", y)
            if [k for k, _ in kx] != [k for k, _ in ky]:
                problems.append("%s keys differ" % pre)
                continue
            for (k, u), (_, v) in zip(kx, ky):
                u, v = float(u), float(v)
                rel = abs(u - v) / max(abs(u), abs(v), 1e-30)
                worst = max(worst, rel)
                if rel > 1e-6:
                    problems.append("%s %s: %r vs %r (rel %.2g)" % (pre, k, u, v, rel))
    v = "PASS" if not problems else "FAIL"
    print("%s L1_gpu_r1_step1  max rel diff of [terms]/[render] = %.3g" % (v, worst))
    for p in problems:
        print("       - " + p)
    return ("L1_gpu_r1_step1", v, problems, 0, "max rel %.3g" % worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="case names (default: all + L1a)")
    ap.add_argument("--gpu", action="store_true", help="run the L1 GPU case only")
    args = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = W + "/runs/variants_verify/l1/" + stamp
    os.makedirs(host(base), exist_ok=True)
    rows = []
    if args.gpu:
        rows.append(gpu_case(base))
        args.only = []

    # the NaN configs: the real test configs with w_pixel = NaN
    import json
    for src, dst in (("esrgan.yaml", "esrgan_nan.yaml"), ("render_r1.yaml", "render_r1_nan.yaml")):
        with open(host(C + "/" + src), encoding="utf-8") as f:
            text = f.read()
        text = re.sub(r"^w_pixel:.*$", "w_pixel: .nan", text, flags=re.M)
        with open(host(C + "/" + dst), "w", encoding="utf-8") as f:
            f.write(text)

    if args.only is None or "L1a_paper" in args.only:
        def up(root, hroot):
            os.makedirs(os.path.join(hroot, "logs"), exist_ok=True)
            return run_side_cwd(UPSTREAM, ["--paper-seed", "3"], root, hroot)

        def ours(root, hroot):
            return run_side_cwd(NEW, ["--paper-seed", "3"], root, hroot, ["--experiment", "paper"])

        def run_side_cwd(script, pre, root, hroot, extra=()):
            code, out, err = docker(["python", DET] + pre + [script] + list(extra) +
                                    ["--cfg_path", C + "/paper.yaml", "--gpu", "-1"], cwd=root)
            for k, v in (("stdout", out), ("stderr", err)):
                with open(os.path.join(hroot, "run0." + k), "w", encoding="utf-8") as f:
                    f.write(v)
            return code, out
        # The harness itself first: upstream against upstream. If this fails,
        # nothing below means anything.
        rows.append(("L1_selfcheck_upstream",) + compare(
            "L1_selfcheck_upstream", up, up, 0, "checkpoints/esrgan_40k", base))
        rows.append(("L1a_paper",) + compare("L1a_paper", up, ours, 0,
                                             "checkpoints/esrgan_40k", base))

    for name, exp, runs, expect, ckpt in CASES:
        if args.only is not None and name not in args.only:
            continue
        old = lambda root, hroot, runs=runs: run_side(OLD, [], runs, root, hroot)
        new = lambda root, hroot, runs=runs, exp=exp: run_side(
            NEW, ["--experiment", exp], runs, root, hroot)
        rows.append((name,) + compare(name, old, new, expect, ckpt, base))

    md = ["# L1 trainer equivalence (%s)" % stamp, "",
          "| case | verdict | stdout lines | checkpoint | problems |", "|---|---|---|---|---|"]
    for name, v, probs, n, ck in rows:
        md.append("| %s | %s | %d | %s | %s |" % (name, v, n, ck.replace("|", "/"),
                                                 "<br>".join(probs).replace("|", "\\|")))
    n_fail = sum(r[1] != "PASS" for r in rows)
    md += ["", "%d cases, %d FAIL" % (len(rows), n_fail)]
    with open(os.path.join(host(base), "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("L1: %d cases, %d FAIL -> %s/report.md" % (len(rows), n_fail, host(base)))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
