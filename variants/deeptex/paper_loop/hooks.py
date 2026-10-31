"""The hook points of loop.py and psnr_loop.py. Every default IS the paper.

loop.py is external/DeepTextures/train_esrgan.py with hook calls inserted, and
psnr_loop.py is the donor's external/esrgan-tf2/train_psnr.py the same way. Each
upstream line a hook call replaced is kept here, verbatim, in the default
implementation or in the comment beside it - l0_skeleton.py checks that every
line upstream.diff removes occurs in this file. Run with these defaults, loop.py
does exactly what train_esrgan.py does (L1a runs the two side by side).

An experiment is a Hooks subclass that overrides some of these:
  fixed/hooks.py         the defect repairs (#2 #3 #4 #5 #8, validation)
  render_loss/hooks.py   grouped batches + MatUp's render term
both built on hooks_common/trainer.py (the old deeptex_train.py's machinery).

Hook                     upstream position (train_esrgan.py)
  on_start(logger)       :24-26   logger off, set_memory_growth
  load_cfg(path)         :28      cfg = load_yaml(FLAGS.cfg_path)
  pre_build(cfg)         before :31 (nothing upstream)
  summary(model)         :32, :34 model.summary
  build_dataset(cfg)     :37      load_dataset(..., shuffle=False)
  build_losses(cfg)      :52-53   GeneratorLoss / DiscriminatorLoss
  checkpoint_dir(cfg)    :56      './checkpoints/' + sub_name
  max_to_keep            :64      3
  restore(...)           :65-81   resume / pretrain lookup
  step_function(fn)      :84      @tf.function
  pre_tape / extra_loss / pre_apply / step_outputs   inside train_step :85-111
  log_dir(cfg)           :115     './logs/' + sub_name
  before_loop(...)       after :117 (nothing upstream)
  batches(ds, n)         :119     train_dataset.take(remain_steps)
  call_step(...)         :123     train_step(lr, hr)
  after_step(...)        :125-128 prog_bar.update
  write_summaries(...)   inside :131 (nothing upstream)
  saved(manager)         :148-149 print save line
  on_end(...)            :151     print training done
"""

from absl import logging
import tensorflow as tf

from modules.losses import DiscriminatorLoss, GeneratorLoss
from modules.utils import load_dataset, load_yaml, set_memory_growth


class Hooks(object):
    """train_esrgan.py's behaviour, one method per replaced line."""

    max_to_keep = 3                               # max_to_keep=3)

    def on_start(self, logger):
        logger.disabled = True
        logger.setLevel(logging.FATAL)
        set_memory_growth()

    def load_cfg(self, cfg_path):
        # cfg = load_yaml(FLAGS.cfg_path)
        return load_yaml(cfg_path)

    def pre_build(self, cfg):
        pass

    def summary(self, model):
        # generator.summary(line_length=80)
        # discriminator.summary(line_length=80)
        model.summary(line_length=80)

    def build_dataset(self, cfg):
        train_dataset = load_dataset(cfg, 'train_dataset', shuffle=False)
        return train_dataset

    def build_losses(self, cfg):
        gen_loss_fn = GeneratorLoss(gan_type=cfg['gan_type'])
        dis_loss_fn = DiscriminatorLoss(gan_type=cfg['gan_type'])
        return gen_loss_fn, dis_loss_fn

    def checkpoint_dir(self, cfg):
        checkpoint_dir = './checkpoints/' + cfg['sub_name']
        return checkpoint_dir

    def restore(self, checkpoint, manager, generator, cfg):
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            print('[*] load ckpt from {} at step {}.'.format(
                manager.latest_checkpoint, checkpoint.step.numpy()))
        else:
            if cfg['pretrain_name'] is not None:
                pretrain_dir = cfg['pretrain_name']
                if tf.train.latest_checkpoint(pretrain_dir):
                    checkpoint.restore(tf.train.latest_checkpoint(pretrain_dir))
                    checkpoint.step.assign(0)
                    print("[*] training from pretrain model {}.".format(
                        pretrain_dir))
                else:
                    print("[*] cannot find pretrain model {}.".format(
                        pretrain_dir))
            else:
                print("[*] training from scratch.")

    def step_function(self, fn):
        # @tf.function
        return tf.function(fn)

    # --- inside train_step. `extras` is empty on the paper path.
    #     def train_step(lr, hr):
    def pre_tape(self, lr, hr, extras):
        return lr, hr, extras

    def extra_loss(self, sr, hr, extras):
        return {}

    def pre_apply(self, grads_G, grads_D):
        """Control dependencies for the two apply_gradients; [] is a no-op."""
        return []

    def step_outputs(self, sr, hr, extras):
        # return total_loss_G, total_loss_D, losses_G, losses_D
        return ()

    def log_dir(self, cfg):
        # './logs/' + cfg['sub_name'])
        return './logs/' + cfg['sub_name']

    def before_loop(self, train_dataset, checkpoint, cfg):
        return train_dataset

    def batches(self, train_dataset, remain_steps):
        # for lr, hr in train_dataset.take(remain_steps):
        return train_dataset.take(remain_steps)

    def call_step(self, train_step, steps, lr, hr, *extras):
        # total_loss_G, total_loss_D, losses_G, losses_D = train_step(lr, hr)
        return train_step(lr, hr, *extras)

    def after_step(self, prog_bar, steps, total_loss_G, total_loss_D, losses_G, losses_D,
                   outs, optimizer_G, optimizer_D):
        prog_bar.update(
            "loss_G={:.4f}, loss_D={:.4f}, lr_G={:.1e}, lr_D={:.1e}".format(
                total_loss_G.numpy(), total_loss_D.numpy(),
                optimizer_G.lr(steps).numpy(), optimizer_D.lr(steps).numpy()))

    def write_summaries(self, steps, outs):
        pass

    def saved(self, manager):
        print("\n[*] save ckpt file at {}".format(
            manager.latest_checkpoint))

    def on_end(self, checkpoint, manager, remain_steps):
        print("\n [*] training done!")

    # def main(_):


class PsnrHooks(Hooks):
    """train_psnr.py's behaviour (the donor's PSNR stage) for psnr_loop.py."""

    # learning_rate = MultiStepLR(cfg['lr'], cfg['lr_steps'], cfg['lr_rate'])
    lr_key = 'lr'

    def summary(self, model):
        # model.summary(line_length=80)
        model.summary(line_length=80)

    def build_dataset(self, cfg):
        train_dataset = load_dataset(cfg, 'train_dataset', shuffle=True)
        return train_dataset

    def restore(self, checkpoint, manager, model, cfg):
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            print('[*] load ckpt from {} at step {}.'.format(
                manager.latest_checkpoint, checkpoint.step.numpy()))
        else:
            print("[*] training from scratch.")

    def call_step(self, train_step, steps, lr, hr, *extras):
        # total_loss, losses = train_step(lr, hr)
        return train_step(lr, hr, *extras)

    def step_outputs(self, sr, hr, extras):
        # return total_loss, losses
        return ()

    def after_step(self, prog_bar, steps, total_loss, losses, outs, optimizer):
        prog_bar.update("loss={:.4f}, lr={:.1e}".format(
            total_loss.numpy(), optimizer.lr(steps).numpy()))

    def on_end(self, checkpoint, manager, remain_steps):
        print("\n[*] training done!")
