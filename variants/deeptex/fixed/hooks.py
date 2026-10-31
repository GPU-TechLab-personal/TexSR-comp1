"""FX - the paper loop with the defects repaired (docs/DEEPTEX-DEVIATIONS.md).

    --experiment fixed --stage psnr     psnr_loop.py  (donor train_psnr.py)
    --experiment fixed --stage esrgan   loop.py       (train_esrgan.py)

What this file changes against the paper, on top of hooks_common/trainer.py:

  #5  build_dataset   shuffle=True (train_esrgan.py:37 passes False; the donor's
                      PSNR stage passes True). --shuffle off restores upstream,
                      which esrgan_asconfigured.yaml needs.
  #8  build_losses    corrected RaGAN, from_logits=True (losses_corrected.py).
                      --losses upstream restores modules/losses.py.
  #2  restore         pretrain_dir (absolute) instead of pretrain_name, proven
                      (hooks_common/ckpt.py) - with trainer.py.
  #3  load_cfg        dead LR schedule refused unless --allow-dead-lr-schedule
                      (hooks_common/safety.check_lr_schedule) - with trainer.py.
  --  validation      --val-every on the held-out split (hooks_common/val.py).

A grouped config (train_dataset.grouped) is the render arm's; this arm refuses it.
"""

from modules.losses import DiscriminatorLoss as UpstreamDLoss
from modules.losses import GeneratorLoss as UpstreamGLoss
from modules.utils import load_dataset

from variants.deeptex.fixed import losses_corrected
from variants.deeptex.hooks_common.safety import die
from variants.deeptex.hooks_common.trainer import (PsnrTrainerHooks, TrainerHooks,
                                                   add_common_args, parse)


def add_args(ap):
    add_common_args(ap)
    ap.add_argument("--stage", required=True, choices=["psnr", "esrgan"])
    ap.add_argument("--losses", default="corrected", choices=["corrected", "upstream"],
                    help="corrected = from_logits=True (see fixed/losses_corrected.py)")
    ap.add_argument("--shuffle", default="on", choices=["on", "off"],
                    help="off reproduces train_esrgan.py:37's shuffle=False")


class _Fixed(object):

    def experiment_cfg(self, cfg):
        a = self.args
        if bool((cfg.get("train_dataset") or {}).get("grouped", False)):
            die(8, "%s is a grouped config; it trains with --experiment render"
                   % a.cfg_path)
        if cfg.get("w_render") or cfg.get("render"):
            die(8, "w_render / render / --overfit-batches / --render-monitor-every need a "
                   "grouped config (train_dataset.grouped: True)")

    def build_dataset(self, cfg):
        train_dataset = load_dataset(cfg, "train_dataset", shuffle=(self.args.shuffle == "on"))
        if self.args.shuffle == "off":
            print("[*] shuffle DISABLED - reproducing train_esrgan.py:37")
        return train_dataset


class FixedHooks(_Fixed, TrainerHooks):

    def build_losses(self, cfg):
        if self.args.losses == "corrected":
            gen_loss_fn = losses_corrected.GeneratorLoss(gan_type=cfg["gan_type"])
            dis_loss_fn = losses_corrected.DiscriminatorLoss(gan_type=cfg["gan_type"])
        else:
            gen_loss_fn = UpstreamGLoss(gan_type=cfg["gan_type"])
            dis_loss_fn = UpstreamDLoss(gan_type=cfg["gan_type"])
        print("[*] adversarial losses: %s" % self.args.losses)
        return gen_loss_fn, dis_loss_fn


class FixedPsnrHooks(_Fixed, PsnrTrainerHooks):
    pass


def build(argv):
    """-> (hooks, loop module name)."""
    args = parse(argv, add_args)
    if args.stage == "psnr":
        return FixedPsnrHooks(args), "psnr_loop"
    return FixedHooks(args), "loop"
