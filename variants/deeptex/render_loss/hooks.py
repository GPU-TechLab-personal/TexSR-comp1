"""RL - the paper loop on grouped batches, plus MatUp's render loss.

    --experiment render   (configs: render_loss/r0a.yaml, r0b.yaml, r1.yaml)

A declared experiment outside the paper, not a repair. One Example holds one
material's five maps (grouped_dataset.py), so a step feeds B materials x 5 maps
= 5B images through the SAME G, D and loss lines as train_esrgan.py's step, plus
w_render x MatUp's l2,1 render loss (render_tf.py) of the regrouped SR stack
against renders of the GT maps. Against the paper, the render arm's changes are
exactly this file plus hooks_common/:

  load_cfg       grouped config checks, seed REQUIRED, w_render resolved to a
                 Python float (traced into the step as a constant)
  build_dataset  none at :37 - the grouped stream is built after the restore,
                 at start_step * batch (resume_dataset), so a resumed run sees
                 the crops the uninterrupted run would have
  build_losses   upstream's RaGAN, unchanged, + RenderLoss after it (it draws no
                 random numbers, but nothing that differs between R0 and R1 may
                 run before the seeded init)
  restore        hooks_common.ckpt (--pretrain-dir for the overfit gate) and the
                 [init] g_sha/d_sha line the smoke gate pairs R0a with R1 by
  step_function  input_signature: one trace for the whole run
  pre_tape       (B,5,..) -> (5B,..) row-major, render target before the tape
  extra_loss     losses_G['render'] = w_render * l2,1   (added after 'gan', so
                 tf.add_n sums reg, pixel, feature, gan, render as before)
  pre_apply      global grad norm asserted finite BEFORE either update
  step_outputs   render diagnostics (mask_frac, oor_*, sr_nz_guard)
  call_step      [render] monitor probe; non-finite -> nan_debug_exit (state and
                 batch to <sub_name>_nan_debug, never the arm's own checkpoints)
  extra_report   [terms] at step 1, every --log-every, and the last step

It runs with upstream's losses, shuffle off and the dead LR schedule allowed -
the settings run_deeptex.ps1 always passed (`--losses upstream --shuffle off
--allow-dead-lr-schedule --pretrain-dir none`). Those flags are still accepted,
with only those values, so the argv stays the same; the fixed arm's repairs are
not part of this experiment.

Checkpoints: max_to_keep stays 3 (upstream, and what the old trainer used).
The 8 checkpoints of a 40k render run are kept by run.ps1's
-KeepAllCheckpoints copier, not by the trainer.
"""

import json
import math
import os
import sys

import numpy as np
import tensorflow as tf

from modules.losses import DiscriminatorLoss as UpstreamDLoss
from modules.losses import GeneratorLoss as UpstreamGLoss

from variants.deeptex.hooks_common import log, safety
from variants.deeptex.hooks_common.ckpt import weights_sha
from variants.deeptex.hooks_common.safety import check_dataset, die
from variants.deeptex.hooks_common.trainer import TrainerHooks, add_common_args, parse
from variants.deeptex.render_loss import grouped_dataset, render_tf, terms


def check_grouped_dataset(cfg, require_verify):
    """check_dataset, plus the two things a grouped record needs on top.

    * meta["schema"] must be grouped_dataset.SCHEMA. The paper-path record has
      the 3-key schema; feeding it to the grouped parser would fail only inside
      tf.data, after the GPU checks and model build.
    * verify_report.json must describe THIS file. check_dataset accepts any PASS
      report beside the record, and the grouped builder writes into a folder
      that can hold an older build's report; the sha256 ties the two together.
    """
    from variants.deeptex.render_loss import grouped_dataset  # lazy in the old trainer; kept

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


def add_args(ap):
    add_common_args(ap)
    ap.add_argument("--stage", default="esrgan", choices=["psnr", "esrgan"],
                    help="a grouped config trains the esrgan step only (psnr: exit 8)")
    ap.add_argument("--losses", default="upstream", choices=["upstream"])
    ap.add_argument("--shuffle", default="off", choices=["off"])
    ap.add_argument("--w-render", type=float, default=None,
                    help="override cfg w_render (the gates pass the calibrated value)")
    ap.add_argument("--overfit-batches", type=int, default=0,
                    help="train on the first N batches only, cached and repeated")
    ap.add_argument("--overfit-fixed", default="16,24,0,0",
                    help="with --overfit-batches: crop 'oy,ox,flip,k' (LR offset, "
                         "flip 0/1, rot90 k) used for every element, or 'none'")
    ap.add_argument("--render-monitor-every", type=int, default=0,
                    help="print '[render] step=N l21=...' (unweighted render l2,1 of "
                         "the current batch) at step 1 and every N steps")


class RenderHooks(TrainerHooks):

    def __init__(self, args):
        TrainerHooks.__init__(self, args)
        self.render_loss, self.render_probe, self.fixed = None, None, None

    # ----------------------------------------------------------------- config
    def experiment_cfg(self, cfg):
        a = self.args
        if not bool((cfg.get("train_dataset") or {}).get("grouped", False)):
            die(8, "w_render / render / --overfit-batches / --render-monitor-every need a "
                   "grouped config (train_dataset.grouped: True)")
        seed = self.seed_value(cfg)
        if a.stage != "esrgan":
            die(8, "a grouped config trains the esrgan step only (got --stage %s)" % a.stage)
        if cfg["ch_size"] != 3:
            die(8, "grouped mode renders RGB maps; ch_size must be 3, got %r" % cfg["ch_size"])
        if seed is None:
            die(8, "grouped mode needs a seed (cfg `seed:` or --seed): it fixes the G/D "
                   "init and keys every crop/flip/rot draw, which is what pairs R0 with R1")
        if a.overfit_batches < 0 or a.render_monitor_every < 0:
            die(8, "--overfit-batches and --render-monitor-every must be >= 0")
        self.w_render = resolve_w_render(cfg, a.w_render)
        if self.w_render > 0 and not cfg.get("render"):
            die(8, "w_render = %g but the config has no `render:` block" % self.w_render)
        try:
            self.render_cfg = render_tf.validate_render_cfg(cfg.get("render"), warn=False)
        except ValueError as e:
            die(8, str(e))
        if a.overfit_batches > 0:
            self.fixed = parse_fixed(a.overfit_fixed)
        # Config roots too, not only the CLI ones: a grouped config is new, and a
        # relative root in it would write into the submodule.
        safety.check_out_root("checkpoint_root",
                              cfg.get("checkpoint_root", "/workspace/runs/deeptex/checkpoints"))
        safety.check_out_root("log_root", cfg.get("log_root", "/workspace/runs/deeptex/logs"))
        fixed = self.fixed
        print("[*] grouped mode: %d materials x %d maps = %d images/step, w_render=%g, "
              "render monitor every %s, seed=%s%s"
              % (cfg["batch_size"], grouped_dataset.N_MAPS,
                 cfg["batch_size"] * grouped_dataset.N_MAPS, self.w_render,
                 a.render_monitor_every or "off", seed,
                 "" if not a.overfit_batches else
                 ", OVERFIT on the first %d batch(es), crop %s"
                 % (a.overfit_batches, "drawn" if fixed is None else
                    ",".join(str(v) for v in fixed))))
        # The step's shapes: B materials x 5 maps, flattened to B*5 images.
        self.n_b, self.n_m = cfg["batch_size"], grouped_dataset.N_MAPS
        self.s_lr, self.s_hr = cfg["input_size"], cfg["gt_size"]
        self.step_spec = [
            tf.TensorSpec([self.n_b, self.n_m, self.s_lr, self.s_lr, 3], tf.float32),  # lr5
            tf.TensorSpec([self.n_b, self.n_m, self.s_hr, self.s_hr, 3], tf.float32),  # hr5
            tf.TensorSpec([self.n_b, self.s_hr, self.s_hr, 2], tf.float32)]            # pos

    def check_dataset(self, cfg):
        check_grouped_dataset(cfg, require_verify=not self.args.skip_verify_gate)

    def build_dataset(self, cfg):
        # The grouped stream starts at the restored step; see resume_dataset.
        print("[*] shuffle DISABLED - reproducing train_esrgan.py:37")
        return None

    def build_losses(self, cfg):
        gen_loss_fn = UpstreamGLoss(gan_type=cfg["gan_type"])
        dis_loss_fn = UpstreamDLoss(gan_type=cfg["gan_type"])
        print("[*] adversarial losses: %s" % self.args.losses)
        if self.w_render > 0 or self.args.render_monitor_every > 0:
            self.render_loss = render_tf.RenderLoss(self.render_cfg)
        return gen_loss_fn, dis_loss_fn

    def restore(self, checkpoint, manager, generator, cfg):
        TrainerHooks.restore(self, checkpoint, manager, generator, cfg)
        # The smoke gate requires R0 and R1 with the same seed to print the same
        # hashes here; that is what "paired" means for their init.
        print("[init] step=%d g_sha=%s d_sha=%s"
              % (int(checkpoint.step.numpy()), weights_sha(generator),
                 weights_sha(checkpoint.discriminator)), flush=True)
        if self.render_loss is not None and self.args.render_monitor_every > 0:
            n_b, n_m, s_lr, s_hr = self.n_b, self.n_m, self.s_lr, self.s_hr
            render_loss = self.render_loss

            # Unweighted render l2,1 of the current batch, in every arm (R0 too).
            # RRDB has no BatchNorm or dropout, so training=False computes the same
            # SR as the step's forward and touches no state.
            @tf.function(input_signature=self.step_spec)
            def render_probe(lr5, hr5, pos):
                sr = generator(tf.reshape(lr5, [n_b * n_m, s_lr, s_lr, 3]), training=False)
                return render_loss.probe(tf.reshape(sr, [n_b, n_m, s_hr, s_hr, 3]), hr5, pos)
            self.render_probe = render_probe

    # ------------------------------------------------------------- the step
    def step_function(self, fn):
        @tf.function(input_signature=self.step_spec)
        def train_step(lr5, hr5, pos):
            return fn(lr5, hr5, pos)
        return train_step

    def pre_tape(self, lr5, hr5, extras):
        # reshape is row-major, so image b*5 + m is map m of material b, and the
        # reshape back to (B, 5, ...) for the render term is its exact inverse.
        (pos,) = extras
        lr = tf.reshape(lr5, [self.n_b * self.n_m, self.s_lr, self.s_lr, 3])
        hr = tf.reshape(hr5, [self.n_b * self.n_m, self.s_hr, self.s_hr, 3])
        gt_disp = valid = None
        if self.w_render > 0:
            # The target holds no variable and is rendered before the
            # persistent tape opens, so none of it is kept for backprop.
            gt_disp, valid = self.render_loss.target(hr5, pos)
        return lr, hr, (hr5, pos, gt_disp, valid)

    def _sr5(self, sr):
        return tf.reshape(sr, [self.n_b, self.n_m, self.s_hr, self.s_hr, 3])

    def extra_loss(self, sr, hr, extras):
        hr5, pos, gt_disp, valid = extras
        if self.w_render > 0:
            return {"render": self.w_render * terms.render_term(
                self.render_loss, self._sr5(sr), pos, gt_disp, valid)}
        return {}

    def pre_apply(self, grads_G, grads_D):
        # A finite loss can carry a non-finite gradient; the esrgan branch would
        # apply it and notice a step later, with the weights already poisoned.
        # Here the step aborts (InvalidArgumentError) before either update.
        grad_norm = tf.linalg.global_norm(grads_G + grads_D)
        return [tf.debugging.assert_all_finite(grad_norm, "non-finite gradient")]

    def step_outputs(self, sr, hr, extras):
        # Diagnostics only (stop_gradient inside); they change no state.
        hr5, pos, gt_disp, valid = extras
        diag = {}
        if self.render_loss is not None:
            diag = self.render_loss.diagnostics(self._sr5(sr))
            if self.w_render == 0:
                valid = self.render_loss.decode(hr5, pos)["valid"]
            # Fraction of GT texels the mask drops from the render mean.
            diag["mask_frac"] = 1.0 - tf.reduce_mean(valid)
        return (diag,)

    # ------------------------------------------------------------- the loop
    def resume_dataset(self, train_dataset, start_step, cfg):
        # Element i of the stream is drawn from key (seed, i), so starting at
        # start_step * batch_size gives a resumed run the crops the uninterrupted
        # run would have seen (upstream restarts at record 0 instead).
        try:
            return grouped_dataset.load_grouped_dataset(
                cfg, shuffle=(self.args.shuffle == "on"), seed=self.seed,
                start_index=int(start_step) * cfg["batch_size"], fixed=self.fixed,
                overfit_batches=self.args.overfit_batches)
        except (ValueError, TypeError) as e:
            die(8, "grouped dataset: %s" % e)

    def call_step(self, train_step, steps, lr, hr, *extras):
        self.saw_step = True
        steps = int(steps)
        batch = (lr, hr) + tuple(extras)
        every = self.args.render_monitor_every
        if self.render_probe is not None and (steps == 1 or steps == self.niter or
                                              steps % every == 0):
            # Measured BEFORE this step's update, i.e. on the generator whose
            # forward the step's losses describe: R1's [terms] render equals
            # w_render x this value, and step 1 is the init R0 and R1 share.
            l21 = float(self.render_probe(*batch).numpy())
            print("[render] step=%d l21=%.6f" % (steps, l21), flush=True)
            with self.writer.as_default():
                tf.summary.scalar("monitor/render_l21", l21, step=steps)
        try:
            out = train_step(*batch)
            g = float(out[0].numpy())
            d = float(out[1].numpy())
        except tf.errors.InvalidArgumentError as e:
            # The finite-gradient assert (or --check-numerics). Anything else
            # is a real bug and keeps its traceback.
            text = str(e)
            if not any(s in text for s in ("non-finite", "NaN", "Inf")):
                raise
            nan_debug_exit(self.checkpoint, self.ckpt_root, self.cfg["sub_name"], steps,
                           batch, nan_reason(text))
        if not (math.isfinite(g) and math.isfinite(d)):
            nan_debug_exit(self.checkpoint, self.ckpt_root, self.cfg["sub_name"], steps,
                           batch, "loss_G=%s loss_D=%s" % (g, d),
                           list(out[2].items()) + list(out[3].items()))
        return out

    def extra_report(self, steps, losses_G, outs):
        (diag,) = outs
        if steps == 1 or steps % self.args.log_every == 0 or steps == self.niter:
            log.terms_line(steps, losses_G, diag)

    def write_summaries(self, steps, outs):
        (diag,) = outs
        for k, v in diag.items():
            tf.summary.scalar("diag/{}".format(k), v, step=steps)


def build(argv):
    return RenderHooks(parse(argv, add_args)), "loop"
