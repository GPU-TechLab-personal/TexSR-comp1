"""What every arm of ours adds to the paper loop, whatever the experiment.

`TrainerHooks` overrides paper_loop.hooks.Hooks with the old deeptex_train.py's
experiment-independent machinery, split out of its 520-line main():

  on_start     pin check, set_memory_growth, --check-numerics; the Python logger
               left at WARNING and the C++ level restored (defect #4, below)
  load_cfg     CLI overrides, output-root checks, seed, gt_size / lr_steps
               assertions, then niter/save_steps overrides
  pre_build    check_gpu (its tf.random.normal is where the old trainer drew
               its first random number), dataset check, disk check
  restore      hooks_common.ckpt (verified resume, proven pretrain, exit 9)
  after_step   [metrics] / NaN -> save + exit 4 / validation
  saved/on_end newline-terminated save lines, final save, empty-dataset exit 5

The experiment hooks (fixed/hooks.py, render_loss/hooks.py) add their own on
top; `add_args` + `argv` give each the old trainer's flags, so run.ps1 passes
the same argv it always did.

Defect #4 (TF_CPP_MIN_LOG_LEVEL): train_esrgan.py:20 sets '3' after
`import tensorflow`, where it does not reach the C++ layer for messages already
configured, and :23-25 silence the PYTHON logger instead. main.py sets '2'
before the import, as deeptex_train.py:77 did; on_start puts that value back
after loop.py's :20 line so nothing logged later is hidden.
"""

import argparse
import math
import os
import sys

import tensorflow as tf

from modules.models import RRDB_Model
from modules.utils import load_yaml, set_memory_growth

from variants.util import paths
from variants.deeptex.paper_loop.hooks import Hooks, PsnrHooks
from variants.deeptex.hooks_common import ckpt, log, safety, seed as seeding, val
from variants.deeptex.hooks_common.safety import die

DEFAULT_CKPT_ROOT = "/workspace/runs/deeptex/checkpoints"
DEFAULT_LOG_ROOT = "/workspace/runs/deeptex/logs"


def add_common_args(ap):
    """deeptex_train.py:441-466's flags, minus the experiment-specific ones."""
    ap.add_argument("--cfg_path", required=True)
    ap.add_argument("--gpu", default="0")
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
    ap.add_argument("--allow-dead-lr-schedule", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--skip-verify-gate", action="store_true")
    ap.add_argument("--check-numerics", action="store_true",
                    help="tf.debugging.enable_check_numerics - slow, for diagnosis only")
    ap.add_argument("--summary", action="store_true",
                    help="print the 300-line model summaries upstream always prints")
    ap.add_argument("--seed", type=int, default=None,
                    help="tf.keras.utils.set_random_seed before the model build")
    ap.add_argument("--checkpoint-root", default="",
                    help="override cfg checkpoint_root (absolute, not under external/)")
    ap.add_argument("--log-root", default="",
                    help="override cfg log_root (absolute, not under external/)")


class TrainerCore(object):
    """Mixed in before Hooks / PsnrHooks. `self.args` is the parsed argv."""

    stage = "esrgan"

    def __init__(self, args):
        self.args = args
        self.tf_cpp_level = os.environ.get("TF_CPP_MIN_LOG_LEVEL")
        self.saw_step = False
        self.window = None
        self.val_pairs, self.val_generator = [], None

    # ----------------------------------------------------------------- startup
    def on_start(self, logger):
        if self.tf_cpp_level is not None:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = self.tf_cpp_level
        paths.check_pin("DeepTextures")
        # NOTE: tf.get_logger() is deliberately left at its default WARNING -
        # upstream silenced this one and left the C++ noise.
        set_memory_growth()
        if self.args.check_numerics:
            tf.debugging.enable_check_numerics()

    def apply_overrides(self, cfg):
        a = self.args
        if a.tfrecord:
            cfg["train_dataset"]["path"] = a.tfrecord
        if a.sub_name:
            cfg["sub_name"] = a.sub_name
        if a.pretrain_dir:
            cfg["pretrain_dir"] = None if a.pretrain_dir == "none" else a.pretrain_dir
        if a.val_hr or a.val_lr:
            cfg.setdefault("val_dataset", {})
            if a.val_hr:
                cfg["val_dataset"]["hr_dir"] = a.val_hr
            if a.val_lr:
                cfg["val_dataset"]["lr_dir"] = a.val_lr
        self.niter = a.max_steps or cfg["niter"]
        self.save_steps = a.save_steps or cfg["save_steps"]
        if a.checkpoint_root:
            safety.check_out_root("--checkpoint-root", a.checkpoint_root)
            cfg["checkpoint_root"] = a.checkpoint_root
        if a.log_root:
            safety.check_out_root("--log-root", a.log_root)
            cfg["log_root"] = a.log_root

    def experiment_cfg(self, cfg):
        """The experiment's own config checks, between the overrides and the seed."""

    def load_cfg(self, cfg_path):
        cfg = load_yaml(cfg_path)
        self.apply_overrides(cfg)
        self.experiment_cfg(cfg)
        self.seed = seeding.apply_seed(self.seed_value(cfg))
        safety.check_gt_size(cfg)
        safety.check_lr_schedule(cfg, self.args.allow_dead_lr_schedule)
        # The paper loop reads niter / save_steps from cfg; from here on they
        # are the overridden values (the config's own niter was checked above).
        cfg["niter"], cfg["save_steps"] = self.niter, self.save_steps
        self.ckpt_root = cfg.get("checkpoint_root", DEFAULT_CKPT_ROOT)
        self.log_root = cfg.get("log_root", DEFAULT_LOG_ROOT)
        self.cfg = cfg
        return cfg

    def seed_value(self, cfg):
        return self.args.seed if self.args.seed is not None else cfg.get("seed")

    def check_dataset(self, cfg):
        safety.check_dataset(cfg, require_verify=not self.args.skip_verify_gate)

    def pre_build(self, cfg):
        safety.check_gpu(self.args.allow_cpu)
        self.check_dataset(cfg)
        os.makedirs(self.ckpt_root, exist_ok=True)
        safety.check_disk(self.ckpt_root, 6.0 if self.stage == "esrgan" else 4.0)

    def summary(self, model):
        if self.args.summary:
            model.summary(line_length=80)

    # ------------------------------------------------------------ checkpoint
    def checkpoint_dir(self, cfg):
        self.ckpt_dir = os.path.join(self.ckpt_root, cfg["sub_name"])
        return self.ckpt_dir

    def restore(self, checkpoint, manager, generator, cfg):
        self.checkpoint, self.manager, self.generator = checkpoint, manager, generator
        ckpt.restore(checkpoint, manager, generator, cfg)

    def log_dir(self, cfg):
        return os.path.join(self.log_root, cfg["sub_name"])

    # ------------------------------------------------------------------- loop
    def setup_val(self, cfg):
        a = self.args
        self.val_pairs = [] if a.no_val else val.load_val_pairs(cfg, a.val_count)
        if not a.no_val and not self.val_pairs:
            print("[warn] no validation tiles found; running blind "
                  "(there is no dataset-level eval anywhere in upstream either)")
        elif self.val_pairs:
            # Separate dynamic-shape twin; weights are copied across before each
            # validation. See the note in validate().
            self.val_generator = RRDB_Model(None, cfg["ch_size"], cfg["network_G"])
            print("[*] validation on %d held-out tile(s) every %d steps"
                  % (len(self.val_pairs), a.val_every))

    def resume_dataset(self, train_dataset, start_step, cfg):
        return train_dataset

    def before_loop(self, train_dataset, checkpoint, cfg):
        self.setup_val(cfg)
        start_step = checkpoint.step.numpy()
        self.remain = max(self.niter - start_step, 0)
        train_dataset = self.resume_dataset(train_dataset, start_step, cfg)
        print("[*] stage=%s sub_name=%s steps %d -> %d  ckpt=%s"
              % (self.stage, cfg["sub_name"], start_step, self.niter, self.ckpt_dir))
        self.window = log.MetricsWindow()
        return train_dataset

    @property
    def writer(self):
        """Our own scalars (val/*, monitor/*) go to the arm's log dir through a
        second writer; loop.py's summary_writer stays upstream's."""
        if getattr(self, "_writer", None) is None:
            self._writer = tf.summary.create_file_writer(self.log_dir(self.cfg))
        return self._writer

    def check_finite(self, steps, g, d, losses_G, losses_D):
        # Free: the .numpy() above already forced a device sync, exactly as
        # upstream's progress bar does at train_esrgan.py:127.
        if not (math.isfinite(g) and math.isfinite(d)):
            print("\n[fatal] non-finite loss at step %d: loss_G=%s loss_D=%s" % (steps, g, d))
            for k, v in list(losses_G.items()) + list(losses_D.items()):
                print("        %-8s %s" % (k, float(v.numpy())))
            self.manager.save()
            print("        debug checkpoint saved at %s" % self.manager.latest_checkpoint)
            sys.exit(4)

    def report(self, steps, g, d, losses_G, losses_D, outs, optimizer_G):
        """Everything after the step, in the old trainer's order."""
        self.window.tick()
        self.check_finite(steps, g, d, losses_G, losses_D)
        if steps % self.args.log_every == 0 or steps == self.niter:
            self.window.line(steps, self.niter, g, d, optimizer_G)
        self.extra_report(steps, losses_G, outs)
        if self.val_pairs and (steps % self.args.val_every == 0 or steps == self.niter):
            self.val_generator.set_weights(self.generator.get_weights())
            v = val.validate(self.val_generator, self.val_pairs, self.cfg["scale"],
                             with_baseline=True)
            with self.writer.as_default():
                tf.summary.scalar("val/psnr", v["psnr"], step=steps)
                tf.summary.scalar("val/ssim", v["ssim"], step=steps)
                if "bicubic_psnr" in v:
                    tf.summary.scalar("val/bicubic_psnr", v["bicubic_psnr"], step=steps)
            print("[val] step=%d n=%d psnr=%.3f ssim=%.4f bicubic_psnr=%.3f delta=%+.3f"
                  % (steps, v["n"], v["psnr"], v["ssim"],
                     v.get("bicubic_psnr", float("nan")),
                     v["psnr"] - v.get("bicubic_psnr", float("nan"))), flush=True)

    def extra_report(self, steps, losses_G, outs):
        pass

    def saved(self, manager):
        print("[*] saved %s" % manager.latest_checkpoint, flush=True)

    def on_end(self, checkpoint, manager, remain_steps):
        final = int(checkpoint.step.numpy())
        if self.saw_step and final % self.save_steps != 0:
            manager.save()
            print("[*] saved %s" % manager.latest_checkpoint)
        if not self.saw_step and self.remain > 0:
            # repeat() of an empty dataset yields nothing; upstream would print
            # "training done!" here and exit 0 having trained nothing at all.
            die(5, "the dataset yielded no batches - the TFRecord is empty or unreadable")
        print("[*] training done at step %d (checkpoint %s)" % (final, manager.latest_checkpoint))


class TrainerHooks(TrainerCore, Hooks):
    """The ESRGAN loop (loop.py) with our machinery."""

    stage = "esrgan"

    def call_step(self, train_step, steps, lr, hr, *extras):
        self.saw_step = True
        return train_step(lr, hr, *extras)

    def after_step(self, prog_bar, steps, total_loss_G, total_loss_D, losses_G, losses_D,
                   outs, optimizer_G, optimizer_D):
        g = float(total_loss_G.numpy())
        d = float(total_loss_D.numpy())
        self.report(int(steps), g, d, losses_G, losses_D, outs, optimizer_G)


class PsnrTrainerHooks(TrainerCore, PsnrHooks):
    """The PSNR loop (psnr_loop.py) with our machinery. The old trainer's psnr
    step returned a constant 0.0 for loss_D and {} for losses_D; [metrics]
    prints the same loss_D=0.0000."""

    stage = "psnr"
    lr_key = "lr_G"                 # the TexSR configs name it lr_G (psnr_texsr.yaml)

    def call_step(self, train_step, steps, lr, hr, *extras):
        self.saw_step = True
        return train_step(lr, hr, *extras)

    def after_step(self, prog_bar, steps, total_loss, losses, outs, optimizer):
        g = float(total_loss.numpy())
        self.report(int(steps), g, 0.0, losses, {}, outs, optimizer)


def parse(argv, add_args):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    return ap.parse_args(argv)
