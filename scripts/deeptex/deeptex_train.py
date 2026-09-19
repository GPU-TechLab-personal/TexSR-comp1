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
"""

import argparse
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

    # --- training step ----------------------------------------------------------
    if is_gan:
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
    print("[*] stage=%s sub_name=%s steps %d -> %d  ckpt=%s"
          % (args.stage, cfg["sub_name"], start_step, niter, checkpoint_dir))
    t_win = time.time()
    win_steps = 0
    saw_step = False

    for lr, hr in train_dataset.take(remain_steps):
        checkpoint.step.assign_add(1)
        steps = int(checkpoint.step.numpy())
        saw_step = True

        total_loss_G, total_loss_D, losses_G, losses_D = train_step(lr, hr)
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
