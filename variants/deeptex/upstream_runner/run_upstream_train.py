#!/usr/bin/env python3
"""Run upstream's train_esrgan.py VERBATIM and reformat its progress bar.

Runs in the deeptex container. This is the "paper" arm: the trainer that executes
is external/DeepTextures/train_esrgan.py itself, unmodified, so the baseline is the
released code rather than a re-implementation claiming equivalence.

WHY A WRAPPER AND NOT A PATCHED TRAINER
---------------------------------------
The only thing missing from upstream's output is newline termination. Its progress
bar (modules/utils.py:113-118) already prints everything the orchestrator parses:

    Training [>>>>] 20/20, loss_G=98.7585, loss_D=0.0012, lr_G=1.0e-04, lr_D=1.0e-04  2.5 step/sec

but writes it with \\r and no newline, so a piped log becomes one enormous line and
run_deeptex.ps1 cannot find a step count in it. Splitting that stream on \\r and
re-emitting selected frames as `[metrics]` lines costs nothing and changes nothing:
this process never touches the training process's weights, gradients, RNG or
optimizer - it only reads its stdout.

So the paper arm gets observability without a single line of upstream being
different from what the authors released. (Upstream never calls tf.random.set_seed
either, so the run is non-deterministic regardless and there is no bit-exact
trajectory to perturb.)

WHAT IS STILL MISSING vs the fixed arm
--------------------------------------
No validation. Upstream has no dataset-level evaluation loop at all, and adding one
INSIDE the run would be a change to the run. Evaluate the periodic checkpoints
instead - it writes one every cfg['save_steps'] - with scripts/deeptex/deeptex_sr.py
from a separate process, after or alongside training.
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.util import paths  # noqa: E402

# The submodule root (DEEPTEX_ROOT override included) comes from util/paths.py.
SUBMODULE = paths.DEEPTEX_ROOT
TRAINER = os.path.join(SUBMODULE, "train_esrgan.py")

# modules/utils.py:116-118 builds this exact string.
PROGRESS = re.compile(
    r"Training \[[^\]]*\]\s+(\d+)/(\d+),\s*"
    r"loss_G=([-\d.eE+]+),\s*loss_D=([-\d.eE+]+),\s*"
    r"lr_G=([-\d.eE+]+),\s*lr_D=([-\d.eE+]+)\s+([\d.]+) step/sec")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg_path", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--log-every", type=int, default=100,
                    help="emit a [metrics] line every N steps (plus the last one)")
    args = ap.parse_args()

    if not os.path.isfile(TRAINER):
        sys.exit("[fatal] upstream trainer not found: %s" % TRAINER)

    env = dict(os.environ)
    # modules/* must import; upstream's own CWD assumption is './checkpoints' and
    # './logs', so the caller's working directory decides where those land. Keep it
    # OUTSIDE the submodule or the checkpoints make `git submodule status` dirty.
    env["PYTHONPATH"] = SUBMODULE + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    print("[*] paper arm: running %s VERBATIM" % TRAINER, flush=True)
    print("[*] cwd=%s (checkpoints/ and logs/ land here)" % os.getcwd(), flush=True)

    proc = subprocess.Popen(
        [sys.executable, TRAINER, "--cfg_path", args.cfg_path, "--gpu", args.gpu],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=0)

    # train_esrgan.py:67-68 prints this when it resumes. If it resumed at or past
    # niter, remain_steps is 0 and the loop never runs - which is "already done",
    # not the empty-dataset failure. Both produce zero steps and upstream's own
    # "training done!" either way, so this line is the only way to tell them apart.
    RESUMED = re.compile(r"load ckpt from .* at step (\d+)")
    resumed_at = None

    buf, last_emitted, final = b"", -1, None
    while True:
        chunk = proc.stdout.read(1)
        if not chunk:
            break
        if chunk in (b"\r", b"\n"):
            line = buf.decode("utf-8", "replace").rstrip()
            buf = b""
            if not line:
                continue
            r = RESUMED.search(line)
            if r:
                resumed_at = int(r.group(1))
            m = PROGRESS.search(line)
            if not m:
                print(line, flush=True)          # pass everything else through
                continue
            step, niter = int(m.group(1)), int(m.group(2))
            final = (step, niter)
            if step == niter or step - last_emitted >= args.log_every or last_emitted < 0:
                last_emitted = step
                print("[metrics] step=%d niter=%d loss_G=%s loss_D=%s lr_G=%s "
                      "steps_per_sec=%s" % (step, niter, m.group(3), m.group(4),
                                            m.group(5), m.group(7)), flush=True)
        else:
            buf += chunk

    rc = proc.wait()
    if buf.strip():
        print(buf.decode("utf-8", "replace").rstrip(), flush=True)
    if final:
        print("[*] upstream trainer finished at step %d/%d" % final, flush=True)
    elif resumed_at is not None:
        # Resumed at or past niter: nothing left to do. Emit a [metrics] line so the
        # orchestrator's "did it reach niter" assertion sees the state it asks about.
        print("[*] already complete: resumed at step %d, nothing to train" % resumed_at,
              flush=True)
        print("[metrics] step=%d niter=%d loss_G=nan loss_D=nan lr_G=nan "
              "steps_per_sec=0.0" % (resumed_at, resumed_at), flush=True)
    else:
        # The 0-record TFRecord case: .repeat() of an empty dataset yields nothing,
        # the loop never runs, and upstream still prints "training done!" and exits 0.
        print("[fatal] upstream trainer produced no training steps at all "
              "(and it did not resume, so the TFRecord is empty)", flush=True)
        sys.exit(5)
    sys.exit(rc)


if __name__ == "__main__":
    main()
