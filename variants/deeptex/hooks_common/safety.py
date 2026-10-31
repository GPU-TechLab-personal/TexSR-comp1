"""Startup checks of the old trainer, moved unchanged (deeptex_train.py:127-232).

Every check runs before a second of GPU time, and each failure is a `[fatal]`
line plus a fixed exit code that run_deeptex.ps1 and the variants run.ps1 files
map to advice (Get-FailureKind). The messages are byte-identical to the old
trainer's, including the old script paths some of them name.

    2  output root relative / under external/     5  dataset
    6  no GPU                                     7  disk
    8  config / flag                              9  pretrain
    4  non-finite loss (render_loss/hooks.py, hooks_common/trainer.py)
"""

import json
import os
import shutil
import sys
import time

import tensorflow as tf

from variants.common import paths

# deeptex_train.py:87-88 derived this from its own location; paths.py owns it now.
EXTERNAL_DIR = paths.EXTERNAL


def die(code, msg):
    print("[fatal] %s" % msg)
    sys.exit(code)


def check_dataset(cfg, require_verify):
    """A truncated TFRecord does not raise on open - TFRecordDataset just ends early,
    and .repeat() turns that into a short silent cycle. The recorded count and hash
    are the only way to notice, so refuse to start without them."""
    path = cfg["train_dataset"]["path"]
    if not os.path.isfile(path):
        die(5, "training TFRecord not found: %s\n"
               "        run texsr_normalize -> texsr_make_pairs -> texsr_build_tfrecord" % path)
    meta_path = path + ".meta.json"
    if not os.path.isfile(meta_path):
        die(5, "no %s - build the TFRecord with texsr_build_tfrecord.py" % meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    n = meta.get("record_count", 0)
    if n <= 0:
        die(5, "TFRecord reports %s records; training would print 'training done!' "
               "in seconds and write no checkpoint" % n)
    if n < cfg["batch_size"]:
        die(5, "%d records < batch_size %d" % (n, cfg["batch_size"]))
    if meta.get("bytes") and os.path.getsize(path) != meta["bytes"]:
        die(5, "TFRecord size %d != meta %d (truncated?)"
               % (os.path.getsize(path), meta["bytes"]))

    report = os.path.join(os.path.dirname(path), "verify_report.json")
    if os.path.isfile(report):
        with open(report) as f:
            verdict = json.load(f).get("verdict")
        if verdict != "PASS":
            die(5, "%s says %s - fix the dataset before training" % (report, verdict))
        print("[*] dataset %d records, verify_report PASS" % n)
    elif require_verify:
        die(5, "no verify_report.json beside the TFRecord; run "
               "texsr_verify_tfrecord.py (or pass --skip-verify-gate)")
    else:
        print("[*] dataset %d records (verify_report absent, gate skipped)" % n)
    return n


def check_gpu(allow_cpu):
    """TF SILENTLY FALLS BACK TO CPU. MatUp cannot - it uses CUDA graphs and dies -
    so run_matup.ps1 gets away with a presence check. Here a missing GPU turns a
    3-hour run into a 3-week one that looks healthy the whole time."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if allow_cpu:
            print("[warn] no GPU visible; --allow-cpu given, continuing")
            return
        die(6, "no GPU visible to TensorFlow. Training would fall back to CPU and "
               "take weeks while looking healthy. Check `docker context show` is "
               "desktop-linux and that compose reserved the device.")
    t0 = time.time()
    with tf.device("/GPU:0"):
        a = tf.random.normal((2048, 2048))
        tf.matmul(a, a).numpy()
        t1 = time.time()
        for _ in range(10):
            c = tf.matmul(a, a)
        c.numpy()
    dt = time.time() - t1
    print("[*] GPU ok: %s, 10x 2048^3 matmul in %.3fs (warmup %.1fs)"
          % (gpus[0].name, dt, t1 - t0))
    if dt > 1.0:
        print("[warn] matmul is slow for an Ampere card - is something else on the GPU?")


def check_disk(path, need_gb):
    free = shutil.disk_usage(path).free / 1e9
    if free < need_gb:
        die(7, "only %.1f GB free at %s, need ~%.1f GB for checkpoints + logs"
               % (free, path, need_gb))
    print("[*] disk: %.1f GB free at %s" % (free, path))


def check_out_root(what, path):
    """Checkpoint/log roots must be absolute and outside external/.

    The container starts in the submodule, so 'runs/...' would create
    external/DeepTextures/runs/..., which its .gitignore does not cover - and the
    submodule would show as modified."""
    if not (path.startswith("/") or os.path.isabs(path)):
        die(2, "%s %r is relative; the deeptex container's working directory is "
               "external/DeepTextures, so give an absolute /workspace/runs/... path"
               % (what, path))
    p = os.path.normpath(path).replace("\\", "/")
    for ext in (EXTERNAL_DIR.replace("\\", "/"), "/workspace/external"):
        if p == ext or p.startswith(ext.rstrip("/") + "/"):
            die(2, "%s %s is inside %s; nothing may be written under external/"
                   % (what, path, ext))


def check_gt_size(cfg):
    if cfg["gt_size"] != cfg["input_size"] * cfg["scale"]:
        die(8, "gt_size %d != input_size %d * scale %d. modules/dataset.py:37 crops "
               "LR to gt_size/scale while train_esrgan.py:31 sizes the generator "
               "from input_size; nothing cross-checks them."
               % (cfg["gt_size"], cfg["input_size"], cfg["scale"]))


def check_lr_schedule(cfg, allow_dead_lr_schedule):
    """Defect #3. Run against the CONFIG's niter, before the --max-steps override
    replaces it: a smoke run is a deliberate truncation and should not have to
    argue with this gate."""
    if cfg["lr_steps"] and cfg["lr_steps"][0] >= cfg["niter"]:
        msg = ("lr_steps[0] = %d >= niter = %d, so the multi-step decay never fires "
               "and the LR is constant for the whole run (OPEN-ITEMS #7)."
               % (cfg["lr_steps"][0], cfg["niter"]))
        if allow_dead_lr_schedule:
            print("[warn] %s Continuing because --allow-dead-lr-schedule." % msg)
        else:
            die(8, msg + "\n        Use configs/deeptex/esrgan_scaled.yaml, or pass "
                         "--allow-dead-lr-schedule to reproduce upstream on purpose.")
