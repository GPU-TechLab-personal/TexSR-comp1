#!/usr/bin/env python3
"""DeepTextures trainer: PSNR pre-train and GAN fine-tune, with the defects fixed.

Runs in the deeptex container. Structurally parallel to
external/DeepTextures/train_esrgan.py so the two stay diffable; every deviation is
one visible hunk with the defect it repairs named beside it.

NOTHING IN external/DeepTextures IS MODIFIED. Its modules are imported and driven
from outside, the arrangement scripts/matup/matup_render.py:14 established for
matup_egsr. That keeps `git submodule status` a one-command answer to "is this
still upstream?", keeps the line numbers in docs/RUNBOOK.md 4 valid, and - the
reason that actually matters - keeps "DeepTextures exactly as published" runnable,
which docs/OPEN-ITEMS.md #6 and #7 both need in order to tell a faithful
reproduction apart from an upstream bug.

Deviations from train_esrgan.py, each keyed to the defect:

  :20-21  TF_CPP_MIN_LOG_LEVEL set AFTER `import tensorflow` at :4, so the C++
          layer never sees it. Set here before the import, where it works.
  :23-25  logger.disabled = True kills the PYTHON logger - which is where
          checkpoint-mismatch and retracing warnings come from - while leaving the
          C++ cuBLAS/NUMA noise that actually motivated it (OPEN-ITEMS #5). The
          wrong one was silenced. Here: C++ at level 2, Python left at WARNING.
  :37     load_dataset(..., shuffle=False). Record order, and therefore batch
          composition, is frozen for the whole run; batch k is the same 8 tiles on
          every one of ~1600 passes. The donor's PSNR stage uses shuffle=True, so
          this looks like an oversight inherited into the GAN stage.
  :66,:73 checkpoint.restore(...) return value discarded - no assert_consumed, no
          expect_partial, so a structurally mismatched checkpoint restores
          partially and silently (and :23-25 removed the warning that would say so).
  :71     pretrain_dir = cfg['pretrain_name'], with pretrain_name './checkpoints/'.
          The donor builds './checkpoints/' + name. So upstream looks for a
          checkpoint state file directly inside ./checkpoints/ and cannot see one
          in ./checkpoints/psnr_pretrain/ - restoring the donor's PSNR stage does
          NOT fix it. Failure is a bare print at :78-79 and training continues from
          random init. Here it is fatal, and the restore is PROVEN by comparing a
          generator weight before and after.
  :119    no numerics check anywhere. With random init (the above) an early D
          saturation is the expected failure mode, and NaN/inf checkpoints get
          written every 5000 steps with nothing to say so.
  :125    ProgressBar writes \\r with no newline, so a piped log becomes one
          enormous line. A newline-terminated [metrics] line is emitted alongside
          it for the orchestrator to parse.
  ---     no dataset-level evaluation exists anywhere in either repo (the donor's
          Set5/Set14 branch was deleted along with test.py's). --val-every adds one
          against the held-out split, which is the only way to learn at hour 1 that
          hour 3 is wasted.

Grouped / render mode (train_dataset.grouped: True, configs/deeptex/
esrgan_matsynth_render_*.yaml) is NOT a repair but a declared experiment
outside the paper: one Example holds one material's five maps
(grouped_dataset.py), so a step feeds B materials x 5 maps = 5B images through
the SAME G, D and loss lines as the esrgan step, plus w_render x MatUp's l2,1
render loss (render_tf.py) of the regrouped SR stack against renders of the GT
maps. --seed fixes the G/D init and the (stateless) crop sequence, so R0 and R1
start from the same weights, see the same crops and differ in w_render (and in
cuDNN's nondeterminism). Gradients are asserted finite BEFORE they are applied
(D's BatchNorm moving statistics still update on a failing batch, but training
never reads them), and a NaN state goes to <sub_name>_nan_debug, never the arm's
own checkpoints.
The esrgan and psnr steps below are byte-for-byte the ones described above.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

# Before importing tensorflow: the C++ layer reads this at import time, which is
# why train_esrgan.py:20 setting it at :20 (after the :4 import) does nothing.
# 2 = hide INFO, keep WARNING and ERROR. Upstream used 3, which hides warnings too.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

SUBMODULE = os.environ.get("DEEPTEX_ROOT", "/workspace/external/DeepTextures")
PINNED_COMMIT = "e4ee562"
# The repo's external/ (= /workspace/external in the container). The deeptex
# service's working_dir is the submodule (docker/compose.yaml), so a relative
# --checkpoint-root would land inside it; check_out_root refuses both.
EXTERNAL_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             os.pardir, os.pardir, "external"))
sys.path.insert(0, SUBMODULE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # deeptex_losses
# texsr_metrics lives in scripts/common because BOTH methods must measure with the
# same code - see the note in that file about runs/montages/ being committed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "common"))                   # texsr_metrics

import tensorflow as tf  # noqa: E402

from modules.models import RRDB_Model, DiscriminatorVGG128  # noqa: E402
from modules.lr_scheduler import MultiStepLR  # noqa: E402
from modules.losses import PixelLoss, ContentLoss  # noqa: E402
from modules.losses import DiscriminatorLoss as UpstreamDLoss  # noqa: E402
from modules.losses import GeneratorLoss as UpstreamGLoss  # noqa: E402
from modules.utils import load_yaml, load_dataset, set_memory_growth  # noqa: E402

import deeptex_losses  # noqa: E402
import texsr_metrics  # noqa: E402


# ------------------------------------------------------------------ startup checks

def check_submodule_pin():
    """Warn loudly if the submodule has moved off the commit our notes describe."""
    try:
        head = subprocess.check_output(
            ["git", "-C", SUBMODULE, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001 - git may be absent inside the container
        return
    if not head.startswith(PINNED_COMMIT):
        print("!" * 72)
        print("! external/DeepTextures is at %s, not the pinned %s." % (head[:12], PINNED_COMMIT))
        print("! docs/DEEPTEX-DEVIATIONS.md and docs/RUNBOOK.md 4 cite line numbers")
        print("! from the pin and may now point at the wrong lines.")
        print("!" * 72)


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


# ------------------------------------------------------------ grouped / render mode

# [terms] names for the step's diagnostics dict, in print order.
#   mask_frac   GT texels dropped from the render mean (invalid GT normal)
#   oor_*       SR values outside [0,1] before the decode clips them (no render
#               gradient there); see render_tf.RenderLoss.diagnostics
#   sr_nz_guard SR texels whose normal the n_z >= nz_min guard moved
TERM_DIAG = (("mask_frac", "mask_frac"), ("oor_basecolor", "oor_bc"),
             ("oor_roughness", "oor_rough"), ("oor_metallic", "oor_metal"),
             ("oor_height", "oor_height"), ("sr_nz_guard", "sr_nz_guard"))


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


def check_grouped_dataset(cfg, require_verify):
    """check_dataset, plus the two things a grouped record needs on top.

    * meta["schema"] must be grouped_dataset.SCHEMA. The paper-path record has
      the 3-key schema; feeding it to the grouped parser would fail only inside
      tf.data, after the GPU checks and model build.
    * verify_report.json must describe THIS file. check_dataset accepts any PASS
      report beside the record, and the grouped builder writes into a folder
      that can hold an older build's report; the sha256 ties the two together.
    """
    import grouped_dataset  # lazy: only the grouped path may depend on it

    path = cfg["train_dataset"]["path"]
    report = os.path.join(os.path.dirname(path), "verify_report.json")
    # check_dataset's own messages name the paper-path tools; say the right ones.
    if not os.path.isfile(path) or not os.path.isfile(path + ".meta.json"):
        die(5, "grouped TFRecord or its .meta.json not found: %s\n"
               "        build it: scripts\\deeptex\\run_deeptex.ps1 -Stage prep-grouped" % path)
    if require_verify and not os.path.isfile(report):
        die(5, "no verify_report.json beside %s; run texsr_verify_grouped.py "
               "(-Stage prep-grouped does), or pass --skip-verify-gate" % path)
    n = check_dataset(cfg, require_verify)
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    if list(meta.get("schema") or []) != list(grouped_dataset.SCHEMA):
        die(5, "%s.meta.json schema is %s, not the grouped schema %s - this is not a "
               "grouped record" % (path, meta.get("schema"), grouped_dataset.SCHEMA))
    if not os.path.isfile(report):
        return n          # only reachable with --skip-verify-gate; check_dataset said so
    with open(report) as f:
        rmeta = json.load(f).get("meta") or {}
    for k in ("sha256", "bytes", "record_count"):
        if rmeta.get(k) != meta.get(k):
            die(5, "%s was written for a different record: %s %r != meta %r.\n"
                   "        Re-run texsr_verify_grouped.py on the current build."
                   % (report, k, rmeta.get(k), meta.get(k)))
    print("[*] grouped record: schema ok, verify_report matches sha256 %s..."
          % str(meta.get("sha256"))[:16])
    return n


def resolve_w_render(cfg, override):
    """The render weight as a Python float (traced into the step as a constant).

    A config that says `w_render: null` has not been calibrated yet, and a run
    started from it would train on a guessed weight. --w-render is for the gates,
    which pass the calibrated value explicitly."""
    if override is not None:
        w = override
    elif "w_render" in cfg:
        if cfg["w_render"] is None:
            die(8, "w_render is null: run -Stage gate-render -Gate calibrate and write "
                   "the value")
        w = cfg["w_render"]
    else:
        w = 0.0
    try:
        w = float(w)          # PyYAML reads an untagged 3e-2 as a string
    except (TypeError, ValueError):
        die(8, "w_render = %r is not a number" % (w,))
    if not math.isfinite(w) or w < 0:
        die(8, "w_render = %r must be finite and >= 0" % (w,))
    return w


def parse_fixed(text):
    """--overfit-fixed 'oy,ox,flip,k' -> tuple of ints, or None for 'none'.
    Range checks happen in grouped_dataset._check_fixed, which knows the limit."""
    if text.strip().lower() == "none":
        return None
    parts = [p.strip() for p in text.split(",")]
    try:
        vals = tuple(int(p) for p in parts)
    except ValueError:
        vals = ()
    if len(vals) != 4:
        die(8, "--overfit-fixed wants 'oy,ox,flip,k' (LR offset, flip 0/1, rot90 k) "
               "or 'none', got %r" % text)
    return vals


def weights_sha(model):
    """First 16 hex of sha256 over every weight (trainable or not), in
    model.get_weights() order. Equal across two runs = identical initial state;
    the smoke gate compares R0 against R1 with it."""
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()[:16]


def nan_reason(text):
    """The informative line of a non-finite InvalidArgumentError from train_step.

    TF 2.10 starts an error raised inside a tf.function with 'Graph execution
    error:' and a 'Detected at node ... defined at' traceback whose quoted source
    lines can themselves contain "non-finite gradient" (the assert's call site),
    so match the kernels' own wording instead: enable_check_numerics' '!!! Detected
    Infinity or NaN in output ...' first, then CheckNumerics' '<msg> : Tensor had
    NaN values' (tf.debugging.assert_all_finite). Last non-empty line otherwise."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for key in ("Detected Infinity or NaN", "Tensor had"):
        for ln in lines:
            if key in ln:
                return ln
    return lines[-1] if lines else "(empty error message)"


def nan_debug_exit(checkpoint, ckpt_root, sub_name, steps, batch, why, terms=None):
    """Grouped mode's NaN exit: state and batch go to <sub_name>_nan_debug.

    The esrgan branch saves through the arm's own manager, which evicts a good
    checkpoint (max_to_keep=3) and becomes latest_checkpoint, so a plain rerun
    resumes from the poisoned state and -KeepAllCheckpoints copies it. Here the
    finite-gradient assert stopped apply_gradients, so the saved weights and Adam
    slots are the state BEFORE step `steps` (its step tensor already says `steps`).
    Exception: D's BatchNorm moving_mean/moving_variance. BN assigns them during
    the forward pass, outside the assert's control dependency, so the failing
    batch has already updated (possibly NaN-poisoned) them. Training never reads
    them - D always runs with training=True - so this only matters to someone
    evaluating D from the debug checkpoint."""
    # Keep this prefix: run_deeptex.ps1 Get-FailureKind maps it to "nan".
    print("\n[fatal] non-finite loss at step %d: %s" % (steps, why))
    for k, v in terms or []:                 # (name, tensor) pairs, G terms then D
        print("        %-8s %s" % (k, float(v.numpy())))
    dbg = os.path.join(ckpt_root, sub_name + "_nan_debug")
    manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=dbg,
                                         max_to_keep=1)
    manager.save()
    print("        debug checkpoint saved at %s (outside %s, so a rerun does not "
          "resume from it)" % (manager.latest_checkpoint, os.path.join(ckpt_root, sub_name)))
    try:
        dump = os.path.join(dbg, "batch_step%d.npz" % steps)
        np.savez_compressed(dump, lr5=batch[0].numpy(), hr5=batch[1].numpy(),
                            pos=batch[2].numpy())
        print("        batch (lr5, hr5, pos) dumped to %s" % dump)
    except Exception as e:  # noqa: BLE001 - the checkpoint above is what matters
        print("        [warn] could not dump the batch: %s" % e)
    sys.exit(4)


# ------------------------------------------------------------------- validation

def load_val_pairs(cfg, limit):
    """Held-out tiles as (name, lr_rgb_uint8, hr_rgb_uint8).

    cv2 gives BGR; training fed RGB via tf.image.decode_png (modules/dataset.py:12-13).
    Converting here is the same bug test.py:57,68,70 has and never fixed - it feeds
    cv2's BGR straight into a model trained on RGB.
    """
    vc = cfg.get("val_dataset") or {}
    hr_dir, lr_dir = vc.get("hr_dir"), vc.get("lr_dir")
    if not hr_dir or not os.path.isdir(hr_dir) or not os.path.isdir(lr_dir or ""):
        return []
    names = sorted(f for f in os.listdir(hr_dir) if f.lower().endswith(".png"))
    if limit:
        # Even stride, so the sample spans sources rather than taking one source's
        # first N tiles.
        step = max(1, len(names) // limit)
        names = names[::step][:limit]
    out = []
    for n in names:
        lr = cv2.imread(os.path.join(lr_dir, n), cv2.IMREAD_UNCHANGED)
        hr = cv2.imread(os.path.join(hr_dir, n), cv2.IMREAD_UNCHANGED)
        if lr is None or hr is None:
            continue
        out.append((n, cv2.cvtColor(lr, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)))
    return out


def validate(generator, pairs, scale, with_baseline):
    """Mean PSNR/SSIM over the held-out tiles, plus a bicubic baseline to beat.

    `generator` must be the DYNAMIC-shape model. The training model is built as
    RRDB_Model(cfg['input_size'], ...) = Input([32, 32, 3]), so Keras rejects a
    120x120 validation tile outright even though the architecture is fully
    convolutional. models.py:98-99 switches to tf.shape(fea) only when size is
    None, which is exactly why test.py:33 builds RRDB_Model(None, ...).
    """
    ps, ss, bps = [], [], []
    for _, lr, hr in pairs:
        x = tf.convert_to_tensor(lr[None].astype(np.float32) / 255.0)
        sr = generator(x, training=False).numpy()[0]
        sr = np.rint(np.clip(sr, 0, 1) * 255).astype(np.uint8)
        h, w = hr.shape[:2]
        sr = sr[:h, :w]
        ps.append(texsr_metrics.psnr(sr, hr))
        ss.append(texsr_metrics.ssim(sr, hr))
        if with_baseline:
            bic = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
            bps.append(texsr_metrics.psnr(bic, hr))
    finite = [p for p in ps if math.isfinite(p)]
    res = {"n": len(ps),
           "psnr": float(np.mean(finite)) if finite else float("nan"),
           "ssim": float(np.mean(ss)) if ss else float("nan")}
    if bps:
        fb = [p for p in bps if math.isfinite(p)]
        res["bicubic_psnr"] = float(np.mean(fb)) if fb else float("nan")
    return res


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg_path", required=True)
    ap.add_argument("--stage", required=True, choices=["psnr", "esrgan"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--losses", default="corrected", choices=["corrected", "upstream"],
                    help="corrected = from_logits=True (see scripts/deeptex/deeptex_losses.py)")
    ap.add_argument("--max-steps", type=int, default=0, help="override niter (smoke runs)")
    ap.add_argument("--save-steps", type=int, default=0, help="override cfg save_steps")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-count", type=int, default=16)
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--tfrecord", default="", help="override train_dataset.path")
    ap.add_argument("--sub-name", default="", help="override cfg sub_name")
    ap.add_argument("--pretrain-dir", default="",
                    help="override cfg pretrain_dir ('none' forces from-scratch)")
    ap.add_argument("--val-hr", default="", help="override val_dataset.hr_dir")
    ap.add_argument("--val-lr", default="", help="override val_dataset.lr_dir")
    ap.add_argument("--shuffle", default="on", choices=["on", "off"],
                    help="off reproduces train_esrgan.py:37's shuffle=False")
    ap.add_argument("--allow-dead-lr-schedule", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--skip-verify-gate", action="store_true")
    ap.add_argument("--check-numerics", action="store_true",
                    help="tf.debugging.enable_check_numerics - slow, for diagnosis only")
    ap.add_argument("--summary", action="store_true",
                    help="print the 300-line model summaries upstream always prints")
    # Grouped / render mode (see the module docstring).
    ap.add_argument("--seed", type=int, default=None,
                    help="tf.keras.utils.set_random_seed before the model build, and the "
                         "key of the grouped crop draws (default: cfg seed; none = unseeded)")
    ap.add_argument("--checkpoint-root", default="",
                    help="override cfg checkpoint_root (absolute, not under external/)")
    ap.add_argument("--log-root", default="",
                    help="override cfg log_root (absolute, not under external/)")
    ap.add_argument("--w-render", type=float, default=None,
                    help="override cfg w_render (the gates pass the calibrated value)")
    ap.add_argument("--overfit-batches", type=int, default=0,
                    help="grouped: train on the first N batches only, cached and repeated")
    ap.add_argument("--overfit-fixed", default="16,24,0,0",
                    help="grouped + --overfit-batches: crop 'oy,ox,flip,k' (LR offset, "
                         "flip 0/1, rot90 k) used for every element, or 'none'")
    ap.add_argument("--render-monitor-every", type=int, default=0,
                    help="grouped: print '[render] step=N l21=...' (unweighted render "
                         "l2,1 of the current batch) at step 1 and every N steps")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    check_submodule_pin()
    # NOTE: tf.get_logger() is deliberately left at its default WARNING. See the
    # module docstring - upstream silenced this one and left the C++ noise.
    set_memory_growth()
    if args.check_numerics:
        tf.debugging.enable_check_numerics()

    cfg = load_yaml(args.cfg_path)
    if args.tfrecord:
        cfg["train_dataset"]["path"] = args.tfrecord
    if args.sub_name:
        cfg["sub_name"] = args.sub_name
    if args.pretrain_dir:
        cfg["pretrain_dir"] = None if args.pretrain_dir == "none" else args.pretrain_dir
    if args.val_hr or args.val_lr:
        cfg.setdefault("val_dataset", {})
        if args.val_hr:
            cfg["val_dataset"]["hr_dir"] = args.val_hr
        if args.val_lr:
            cfg["val_dataset"]["lr_dir"] = args.val_lr
    niter = args.max_steps or cfg["niter"]
    save_steps = args.save_steps or cfg["save_steps"]
    is_gan = args.stage == "esrgan"
    if args.checkpoint_root:
        check_out_root("--checkpoint-root", args.checkpoint_root)
        cfg["checkpoint_root"] = args.checkpoint_root
    if args.log_root:
        check_out_root("--log-root", args.log_root)
        cfg["log_root"] = args.log_root

    # --- grouped / render mode: settled here, before any GPU time ---------------
    # grouped_dataset and render_tf are imported only on this path, so the
    # textures_db and PSNR runs never depend on them.
    grouped = bool((cfg.get("train_dataset") or {}).get("grouped", False))
    seed = args.seed if args.seed is not None else cfg.get("seed")
    w_render, render_cfg, fixed = 0.0, None, None
    if grouped:
        import grouped_dataset
        import render_tf
        if not is_gan:
            die(8, "a grouped config trains the esrgan step only (got --stage %s)" % args.stage)
        if cfg["ch_size"] != 3:
            die(8, "grouped mode renders RGB maps; ch_size must be 3, got %r" % cfg["ch_size"])
        if seed is None:
            die(8, "grouped mode needs a seed (cfg `seed:` or --seed): it fixes the G/D "
                   "init and keys every crop/flip/rot draw, which is what pairs R0 with R1")
        if args.overfit_batches < 0 or args.render_monitor_every < 0:
            die(8, "--overfit-batches and --render-monitor-every must be >= 0")
        w_render = resolve_w_render(cfg, args.w_render)
        if w_render > 0 and not cfg.get("render"):
            die(8, "w_render = %g but the config has no `render:` block" % w_render)
        try:
            render_cfg = render_tf.validate_render_cfg(cfg.get("render"), warn=False)
        except ValueError as e:
            die(8, str(e))
        if args.overfit_batches > 0:
            fixed = parse_fixed(args.overfit_fixed)
        # Config roots too, not only the CLI ones: a grouped config is new, and a
        # relative root in it would write into the submodule.
        check_out_root("checkpoint_root",
                       cfg.get("checkpoint_root", "/workspace/runs/deeptex/checkpoints"))
        check_out_root("log_root", cfg.get("log_root", "/workspace/runs/deeptex/logs"))
        print("[*] grouped mode: %d materials x %d maps = %d images/step, w_render=%g, "
              "render monitor every %s, seed=%s%s"
              % (cfg["batch_size"], grouped_dataset.N_MAPS,
                 cfg["batch_size"] * grouped_dataset.N_MAPS, w_render,
                 args.render_monitor_every or "off", seed,
                 "" if not args.overfit_batches else
                 ", OVERFIT on the first %d batch(es), crop %s"
                 % (args.overfit_batches, "drawn" if fixed is None else
                    ",".join(str(v) for v in fixed))))
    elif (args.w_render or cfg.get("w_render") or cfg.get("render")
          or args.overfit_batches or args.render_monitor_every):
        die(8, "w_render / render / --overfit-batches / --render-monitor-every need a "
               "grouped config (train_dataset.grouped: True)")
    if seed is not None:
        # Before check_gpu's random matmul and the model build. tf.random.set_seed
        # alone does NOT fix the init under Keras 2.10: VarianceScaling (every G/D
        # conv, modules/models.py:11-15) and glorot_uniform (D's Dense) take their
        # stateless seed from backend._SEED_GENERATOR, else from Python's unseeded
        # global `random`. set_random_seed seeds random, np.random, tf.random AND
        # that generator: identical G/D init in every run with this seed.
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            die(8, "seed = %r is not an integer" % (seed,))
        if not 0 <= seed < 2 ** 32:
            die(8, "seed = %d is outside [0, 2**32), which np.random.seed rejects" % seed)
        tf.keras.utils.set_random_seed(seed)
        print("[*] tf.keras.utils.set_random_seed(%d)" % seed)

    # --- startup assertions, all before a single second of GPU time -------------
    if cfg["gt_size"] != cfg["input_size"] * cfg["scale"]:
        die(8, "gt_size %d != input_size %d * scale %d. modules/dataset.py:37 crops "
               "LR to gt_size/scale while train_esrgan.py:31 sizes the generator "
               "from input_size; nothing cross-checks them."
               % (cfg["gt_size"], cfg["input_size"], cfg["scale"]))
    # Validate the CONFIG, not the --max-steps override: a smoke run is a
    # deliberate truncation and should not have to argue with this gate.
    if cfg["lr_steps"] and cfg["lr_steps"][0] >= cfg["niter"]:
        msg = ("lr_steps[0] = %d >= niter = %d, so the multi-step decay never fires "
               "and the LR is constant for the whole run (OPEN-ITEMS #7)."
               % (cfg["lr_steps"][0], cfg["niter"]))
        if args.allow_dead_lr_schedule:
            print("[warn] %s Continuing because --allow-dead-lr-schedule." % msg)
        else:
            die(8, msg + "\n        Use configs/deeptex/esrgan_scaled.yaml, or pass "
                         "--allow-dead-lr-schedule to reproduce upstream on purpose.")
    check_gpu(args.allow_cpu)
    if grouped:
        check_grouped_dataset(cfg, require_verify=not args.skip_verify_gate)
    else:
        check_dataset(cfg, require_verify=not args.skip_verify_gate)
    ckpt_root = cfg.get("checkpoint_root", "/workspace/runs/deeptex/checkpoints")
    log_root = cfg.get("log_root", "/workspace/runs/deeptex/logs")
    os.makedirs(ckpt_root, exist_ok=True)
    check_disk(ckpt_root, 6.0 if is_gan else 4.0)

    # --- define network ---------------------------------------------------------
    generator = RRDB_Model(cfg["input_size"], cfg["ch_size"], cfg["network_G"])
    if args.summary:
        generator.summary(line_length=80)
    discriminator = None
    if is_gan:
        # nf comes from the Python default; cfg['network_D'] is a dead key in both
        # repos (grep it - nothing reads it).
        discriminator = DiscriminatorVGG128(cfg["gt_size"], cfg["ch_size"])
        if args.summary:
            discriminator.summary(line_length=80)

    # --- load dataset -----------------------------------------------------------
    # shuffle=True by default. Upstream's GAN stage passes False
    # (train_esrgan.py:37) while the donor's PSNR stage passes True, freezing batch
    # composition for the whole run. --shuffle off restores upstream's behaviour,
    # which the "as configured" baseline needs in order to be honest: without it,
    # esrgan_asconfigured.yaml would reproduce only 2 of upstream's 4 deviations.
    # Grouped mode builds its loader after the checkpoint restore instead: the
    # stream resumes at the restored step, so that step must be known first.
    if not grouped:
        train_dataset = load_dataset(cfg, "train_dataset", shuffle=(args.shuffle == "on"))
    if args.shuffle == "off":
        print("[*] shuffle DISABLED - reproducing train_esrgan.py:37")

    # --- define optimizer -------------------------------------------------------
    lr_G = MultiStepLR(cfg["lr_G"], cfg["lr_steps"], cfg["lr_rate"])
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=lr_G,
                                           beta_1=cfg["adam_beta1_G"],
                                           beta_2=cfg["adam_beta2_G"])
    optimizer_D = None
    if is_gan:
        lr_D = MultiStepLR(cfg["lr_D"], cfg["lr_steps"], cfg["lr_rate"])
        optimizer_D = tf.keras.optimizers.Adam(learning_rate=lr_D,
                                               beta_1=cfg["adam_beta1_D"],
                                               beta_2=cfg["adam_beta2_D"])

    # --- define losses function -------------------------------------------------
    pixel_loss_fn = PixelLoss(criterion=cfg["pixel_criterion"])
    if is_gan:
        fea_loss_fn = ContentLoss(criterion=cfg["feature_criterion"])
        if args.losses == "corrected":
            gen_loss_fn = deeptex_losses.GeneratorLoss(gan_type=cfg["gan_type"])
            dis_loss_fn = deeptex_losses.DiscriminatorLoss(gan_type=cfg["gan_type"])
        else:
            gen_loss_fn = UpstreamGLoss(gan_type=cfg["gan_type"])
            dis_loss_fn = UpstreamDLoss(gan_type=cfg["gan_type"])
        print("[*] adversarial losses: %s" % args.losses)
    # Built after G and D on purpose: it draws no random numbers, but nothing that
    # differs between R0 and R1 may run before the seeded init.
    render_loss = None
    if grouped and (w_render > 0 or args.render_monitor_every > 0):
        render_loss = render_tf.RenderLoss(render_cfg)

    # --- load checkpoint --------------------------------------------------------
    checkpoint_dir = os.path.join(ckpt_root, cfg["sub_name"])
    ckpt_kwargs = {"step": tf.Variable(0, name="step"), "model": generator}
    if is_gan:
        ckpt_kwargs.update(optimizer_G=optimizer_G, optimizer_D=optimizer_D,
                           discriminator=discriminator)
    else:
        # Matches the donor's train_psnr.py:47-49 structure, so an ESRGAN
        # Checkpoint can later restore 'model' out of it.
        ckpt_kwargs.update(optimizer=optimizer_G)
    checkpoint = tf.train.Checkpoint(**ckpt_kwargs)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir, max_to_keep=3)

    if manager.latest_checkpoint:
        status = checkpoint.restore(manager.latest_checkpoint)
        status.assert_existing_objects_matched()   # upstream discards this
        print("[*] resumed from %s at step %d."
              % (manager.latest_checkpoint, checkpoint.step.numpy()))
    elif cfg.get("pretrain_dir"):
        pretrain_dir = cfg["pretrain_dir"]
        latest = tf.train.latest_checkpoint(pretrain_dir)
        if not latest:
            # Upstream prints and carries on from random init. That is the single
            # biggest unknown in OPEN-ITEMS #6, and it is silent. Not here.
            die(9, "pretrain_dir is set to %s but no checkpoint is there.\n"
                   "        Run --stage psnr first, or set pretrain_dir: null to "
                   "train from scratch on purpose." % pretrain_dir)
        before = generator.trainable_variables[0].numpy().copy()
        checkpoint.restore(latest).expect_partial()   # PSNR ckpt has no D / optimizer_G
        after = generator.trainable_variables[0].numpy()
        if np.array_equal(before, after):
            die(9, "restoring %s did not change the generator's weights.\n"
                   "        The checkpoint resolved but nothing was applied - this "
                   "is exactly the silent no-op OPEN-ITEMS #6 describes." % latest)
        delta = float(np.abs(after - before).max())
        checkpoint.step.assign(0)
        print("[*] pretrain restored from %s (max |dW| = %.4g on '%s')"
              % (latest, delta, generator.trainable_variables[0].name))
    else:
        print("[*] training from scratch (pretrain_dir is null).")
    if grouped:
        # The smoke gate requires R0 and R1 with the same seed to print the same
        # hashes here; that is what "paired" means for their init.
        print("[init] step=%d g_sha=%s d_sha=%s"
              % (int(checkpoint.step.numpy()), weights_sha(generator),
                 weights_sha(discriminator)), flush=True)

    # --- training step ----------------------------------------------------------
    render_probe = None
    if grouped:
        # The esrgan step below on B materials x 5 maps, flattened to B*5 images.
        # reshape is row-major, so image b*5 + m is map m of material b, and the
        # reshape back to (B, 5, ...) for the render term is its exact inverse.
        # input_signature + drop_remainder: one trace for the whole run.
        n_b, n_m = cfg["batch_size"], grouped_dataset.N_MAPS
        s_lr, s_hr = cfg["input_size"], cfg["gt_size"]
        step_spec = [tf.TensorSpec([n_b, n_m, s_lr, s_lr, 3], tf.float32),    # lr5
                     tf.TensorSpec([n_b, n_m, s_hr, s_hr, 3], tf.float32),    # hr5
                     tf.TensorSpec([n_b, s_hr, s_hr, 2], tf.float32)]         # pos

        @tf.function(input_signature=step_spec)
        def train_step(lr5, hr5, pos):
            lr = tf.reshape(lr5, [n_b * n_m, s_lr, s_lr, 3])
            hr = tf.reshape(hr5, [n_b * n_m, s_hr, s_hr, 3])
            if w_render > 0:
                # The target holds no variable and is rendered before the
                # persistent tape opens, so none of it is kept for backprop.
                gt_disp, valid = render_loss.target(hr5, pos)
            with tf.GradientTape(persistent=True) as tape:
                sr = generator(lr, training=True)
                hr_output = discriminator(hr, training=True)
                sr_output = discriminator(sr, training=True)

                losses_G, losses_D = {}, {}
                losses_G["reg"] = tf.reduce_sum(generator.losses)
                losses_D["reg"] = tf.reduce_sum(discriminator.losses)
                losses_G["pixel"] = cfg["w_pixel"] * pixel_loss_fn(hr, sr)
                losses_G["feature"] = cfg["w_feature"] * fea_loss_fn(hr, sr)
                losses_G["gan"] = cfg["w_gan"] * gen_loss_fn(hr_output, sr_output)
                losses_D["gan"] = dis_loss_fn(hr_output, sr_output)
                if w_render > 0:
                    losses_G["render"] = w_render * render_loss.loss(
                        tf.reshape(sr, [n_b, n_m, s_hr, s_hr, 3]), pos, gt_disp, valid)
                total_loss_G = tf.add_n([v for v in losses_G.values()])
                total_loss_D = tf.add_n([v for v in losses_D.values()])

            grads_G = tape.gradient(total_loss_G, generator.trainable_variables)
            grads_D = tape.gradient(total_loss_D, discriminator.trainable_variables)
            # A finite loss can carry a non-finite gradient; the esrgan branch would
            # apply it and notice a step later, with the weights already poisoned.
            # Here the step aborts (InvalidArgumentError) before either update.
            grad_norm = tf.linalg.global_norm(grads_G + grads_D)
            with tf.control_dependencies(
                    [tf.debugging.assert_all_finite(grad_norm, "non-finite gradient")]):
                optimizer_G.apply_gradients(zip(grads_G, generator.trainable_variables))
                optimizer_D.apply_gradients(zip(grads_D, discriminator.trainable_variables))

            # Diagnostics only (stop_gradient inside); they change no state.
            diag = {}
            if render_loss is not None:
                diag = render_loss.diagnostics(tf.reshape(sr, [n_b, n_m, s_hr, s_hr, 3]))
                if w_render == 0:
                    valid = render_loss.decode(hr5, pos)["valid"]
                # Fraction of GT texels the mask drops from the render mean.
                diag["mask_frac"] = 1.0 - tf.reduce_mean(valid)
            return total_loss_G, total_loss_D, losses_G, losses_D, diag

        if render_loss is not None and args.render_monitor_every > 0:
            # Unweighted render l2,1 of the current batch, in every arm (R0 too).
            # RRDB has no BatchNorm or dropout, so training=False computes the same
            # SR as the step's forward and touches no state.
            @tf.function(input_signature=step_spec)
            def render_probe(lr5, hr5, pos):
                sr = generator(tf.reshape(lr5, [n_b * n_m, s_lr, s_lr, 3]), training=False)
                return render_loss.probe(tf.reshape(sr, [n_b, n_m, s_hr, s_hr, 3]), hr5, pos)

    elif is_gan:
        @tf.function
        def train_step(lr, hr):
            with tf.GradientTape(persistent=True) as tape:
                sr = generator(lr, training=True)
                hr_output = discriminator(hr, training=True)
                sr_output = discriminator(sr, training=True)

                losses_G, losses_D = {}, {}
                losses_G["reg"] = tf.reduce_sum(generator.losses)
                losses_D["reg"] = tf.reduce_sum(discriminator.losses)
                losses_G["pixel"] = cfg["w_pixel"] * pixel_loss_fn(hr, sr)
                losses_G["feature"] = cfg["w_feature"] * fea_loss_fn(hr, sr)
                losses_G["gan"] = cfg["w_gan"] * gen_loss_fn(hr_output, sr_output)
                losses_D["gan"] = dis_loss_fn(hr_output, sr_output)
                total_loss_G = tf.add_n([v for v in losses_G.values()])
                total_loss_D = tf.add_n([v for v in losses_D.values()])

            grads_G = tape.gradient(total_loss_G, generator.trainable_variables)
            grads_D = tape.gradient(total_loss_D, discriminator.trainable_variables)
            optimizer_G.apply_gradients(zip(grads_G, generator.trainable_variables))
            optimizer_D.apply_gradients(zip(grads_D, discriminator.trainable_variables))
            return total_loss_G, total_loss_D, losses_G, losses_D
    else:
        @tf.function
        def train_step(lr, hr):
            with tf.GradientTape() as tape:
                sr = generator(lr, training=True)
                losses_G = {"reg": tf.reduce_sum(generator.losses),
                            "pixel": cfg["w_pixel"] * pixel_loss_fn(hr, sr)}
                total_loss_G = tf.add_n([v for v in losses_G.values()])
            grads_G = tape.gradient(total_loss_G, generator.trainable_variables)
            optimizer_G.apply_gradients(zip(grads_G, generator.trainable_variables))
            return total_loss_G, tf.constant(0.0), losses_G, {}

    # --- training loop ----------------------------------------------------------
    summary_writer = tf.summary.create_file_writer(os.path.join(log_root, cfg["sub_name"]))
    val_pairs = [] if args.no_val else load_val_pairs(cfg, args.val_count)
    val_generator = None
    if not args.no_val and not val_pairs:
        print("[warn] no validation tiles found; running blind "
              "(there is no dataset-level eval anywhere in upstream either)")
    elif val_pairs:
        # Separate dynamic-shape twin; weights are copied across before each
        # validation. See the note in validate().
        val_generator = RRDB_Model(None, cfg["ch_size"], cfg["network_G"])
        print("[*] validation on %d held-out tile(s) every %d steps"
              % (len(val_pairs), args.val_every))

    start_step = checkpoint.step.numpy()
    remain_steps = max(niter - start_step, 0)
    if grouped:
        # Element i of the stream is drawn from key (seed, i), so starting at
        # start_step * batch_size gives a resumed run the crops the uninterrupted
        # run would have seen (upstream restarts at record 0 instead).
        try:
            train_dataset = grouped_dataset.load_grouped_dataset(
                cfg, shuffle=(args.shuffle == "on"), seed=seed,
                start_index=int(start_step) * cfg["batch_size"], fixed=fixed,
                overfit_batches=args.overfit_batches)
        except (ValueError, TypeError) as e:
            die(8, "grouped dataset: %s" % e)
    print("[*] stage=%s sub_name=%s steps %d -> %d  ckpt=%s"
          % (args.stage, cfg["sub_name"], start_step, niter, checkpoint_dir))
    t_win = time.time()
    win_steps = 0
    saw_step = False

    for batch in train_dataset.take(remain_steps):
        checkpoint.step.assign_add(1)
        steps = int(checkpoint.step.numpy())
        saw_step = True

        if grouped:
            if render_probe is not None and (steps == 1 or steps == niter or
                                             steps % args.render_monitor_every == 0):
                # Measured BEFORE this step's update, i.e. on the generator whose
                # forward the step's losses describe: R1's [terms] render equals
                # w_render x this value, and step 1 is the init R0 and R1 share.
                l21 = float(render_probe(*batch).numpy())
                print("[render] step=%d l21=%.6f" % (steps, l21), flush=True)
                with summary_writer.as_default():
                    tf.summary.scalar("monitor/render_l21", l21, step=steps)
            try:
                total_loss_G, total_loss_D, losses_G, losses_D, diag = train_step(*batch)
                g = float(total_loss_G.numpy())
                d = float(total_loss_D.numpy())
            except tf.errors.InvalidArgumentError as e:
                # The finite-gradient assert (or --check-numerics). Anything else
                # is a real bug and keeps its traceback.
                text = str(e)
                if not any(s in text for s in ("non-finite", "NaN", "Inf")):
                    raise
                nan_debug_exit(checkpoint, ckpt_root, cfg["sub_name"], steps, batch,
                               nan_reason(text))
            if not (math.isfinite(g) and math.isfinite(d)):
                nan_debug_exit(checkpoint, ckpt_root, cfg["sub_name"], steps, batch,
                               "loss_G=%s loss_D=%s" % (g, d),
                               list(losses_G.items()) + list(losses_D.items()))
        else:
            total_loss_G, total_loss_D, losses_G, losses_D = train_step(*batch)
            g = float(total_loss_G.numpy())
            d = float(total_loss_D.numpy())
        win_steps += 1

        # Free: the .numpy() above already forced a device sync, exactly as
        # upstream's progress bar does at train_esrgan.py:127.
        if not (math.isfinite(g) and math.isfinite(d)):
            print("\n[fatal] non-finite loss at step %d: loss_G=%s loss_D=%s" % (steps, g, d))
            for k, v in list(losses_G.items()) + list(losses_D.items()):
                print("        %-8s %s" % (k, float(v.numpy())))
            manager.save()
            print("        debug checkpoint saved at %s" % manager.latest_checkpoint)
            sys.exit(4)

        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar("loss_G/total_loss", total_loss_G, step=steps)
                for k, v in losses_G.items():
                    tf.summary.scalar("loss_G/{}".format(k), v, step=steps)
                if is_gan:
                    tf.summary.scalar("loss_D/total_loss", total_loss_D, step=steps)
                    for k, v in losses_D.items():
                        tf.summary.scalar("loss_D/{}".format(k), v, step=steps)
                    tf.summary.scalar("learning_rate_D", optimizer_D.lr(steps), step=steps)
                tf.summary.scalar("learning_rate_G", optimizer_G.lr(steps), step=steps)
        if grouped and diag and steps % 10 == 0:
            with summary_writer.as_default():
                for k, v in diag.items():
                    tf.summary.scalar("diag/{}".format(k), v, step=steps)

        if steps % args.log_every == 0 or steps == niter:
            dt = time.time() - t_win
            sps = win_steps / dt if dt > 0 else 0.0
            t_win, win_steps = time.time(), 0
            mem = tf.config.experimental.get_memory_info("GPU:0")["peak"] // (1 << 20) \
                if tf.config.list_physical_devices("GPU") else 0
            eta = (niter - steps) / sps / 3600.0 if sps > 0 else float("nan")
            # Newline-terminated and machine-readable. run_deeptex.ps1 parses THIS,
            # never the \r progress bar.
            print("[metrics] step=%d niter=%d loss_G=%.4f loss_D=%.4f lr_G=%.2e "
                  "steps_per_sec=%.2f gpu_mem_mb=%d eta_h=%.2f"
                  % (steps, niter, g, d, float(optimizer_G.lr(steps).numpy()),
                     sps, mem, eta), flush=True)
        if grouped and (steps == 1 or steps % args.log_every == 0 or steps == niter):
            # The weighted G terms one by one ([metrics] only has their sum), then
            # the render diagnostics when a RenderLoss exists.
            parts = ["%s=%.5g" % (k, float(losses_G[k].numpy()))
                     for k in ("reg", "pixel", "feature", "gan", "render") if k in losses_G]
            parts += ["%s=%.5g" % (short, float(diag[k].numpy()))
                      for k, short in TERM_DIAG if k in diag]
            print("[terms] step=%d %s" % (steps, " ".join(parts)), flush=True)

        if val_pairs and (steps % args.val_every == 0 or steps == niter):
            val_generator.set_weights(generator.get_weights())
            v = validate(val_generator, val_pairs, cfg["scale"], with_baseline=True)
            with summary_writer.as_default():
                tf.summary.scalar("val/psnr", v["psnr"], step=steps)
                tf.summary.scalar("val/ssim", v["ssim"], step=steps)
                if "bicubic_psnr" in v:
                    tf.summary.scalar("val/bicubic_psnr", v["bicubic_psnr"], step=steps)
            print("[val] step=%d n=%d psnr=%.3f ssim=%.4f bicubic_psnr=%.3f delta=%+.3f"
                  % (steps, v["n"], v["psnr"], v["ssim"],
                     v.get("bicubic_psnr", float("nan")),
                     v["psnr"] - v.get("bicubic_psnr", float("nan"))), flush=True)

        if steps % save_steps == 0:
            manager.save()
            print("[*] saved %s" % manager.latest_checkpoint, flush=True)

    final = int(checkpoint.step.numpy())
    if saw_step and final % save_steps != 0:
        manager.save()
        print("[*] saved %s" % manager.latest_checkpoint)

    if not saw_step and remain_steps > 0:
        # repeat() of an empty dataset yields nothing; upstream would print
        # "training done!" here and exit 0 having trained nothing at all.
        die(5, "the dataset yielded no batches - the TFRecord is empty or unreadable")

    print("[*] training done at step %d (checkpoint %s)" % (final, manager.latest_checkpoint))


if __name__ == "__main__":
    main()
