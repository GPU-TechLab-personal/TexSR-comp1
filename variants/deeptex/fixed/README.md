# fixed/ - FX: the paper loop with the defects repaired

`docs/DEEPTEX-DEVIATIONS.md`'s fixed arm: PSNR pre-train, then ESRGAN from it.

    python variants/deeptex/paper_loop/main.py --experiment fixed --stage psnr   --cfg_path /workspace/variants/deeptex/fixed/psnr_texsr.yaml ...
    python variants/deeptex/paper_loop/main.py --experiment fixed --stage esrgan --cfg_path /workspace/variants/deeptex/fixed/esrgan_scaled.yaml ...

`--stage psnr` runs `paper_loop/psnr_loop.py` (the donor's `train_psnr.py`
with hooks), `--stage esrgan` runs `paper_loop/loop.py` (`train_esrgan.py`).
Every flag is the old `scripts/deeptex/deeptex_train.py`'s.

## What this arm changes against the paper

| # | defect (DEEPTEX-DEVIATIONS) | upstream | here |
|---|---|---|---|
| #2 | pretrain lookup | `train_esrgan.py:71` `pretrain_dir = cfg['pretrain_name']` = `'./checkpoints/'`, misses `./checkpoints/psnr_pretrain/`, prints and trains from random init | `pretrain_dir` (absolute), exit 9 when nothing is there, restore **proven** by a weight delta (`hooks_common/ckpt.py`) |
| #3 | dead LR schedule | `lr_steps[0] = 50000 >= niter = 40000` | refused (exit 8) unless `--allow-dead-lr-schedule` (`hooks_common/safety.check_lr_schedule`); `esrgan_scaled.yaml` has a live one |
| #4 | log levels | `:20` C++ level set after the import; `:23-25` Python logger silenced | C++ level 2 before the import, Python logger at WARNING (`main.py`, `hooks_common/trainer.on_start`) |
| #5 | frozen batch order | `:37` `shuffle=False` | `shuffle=True`; `--shuffle off` restores (`hooks.py build_dataset`) |
| #8 | RaGAN on sigmoids | `modules/losses.py:56,80` `from_logits=False` after a manual sigmoid | `losses_corrected.py`, `from_logits=True`; `--losses upstream` restores (`hooks.py build_losses`) |
| - | restore unchecked | `:66,:73` status discarded | `assert_existing_objects_matched()` on resume |
| - | no numerics check | - | non-finite loss → `[fatal] non-finite loss`, save, exit 4 |
| - | no validation | - | `--val-every` on the held-out split (`hooks_common/val.py`) |

## Configs

`psnr_texsr.yaml`, `esrgan_scaled.yaml`, `esrgan_asconfigured.yaml`: copied
byte for byte from `configs/deeptex/`. `esrgan_asconfigured.yaml` is upstream's
recipe through this trainer (`--losses upstream --shuffle off
--allow-dead-lr-schedule`, `pretrain_dir: null`).

**Kept as it was:** the old `run_deeptex.ps1` has no stage that runs
`esrgan_asconfigured.yaml` (only `-Config` would reach it); the new `run.ps1`
keeps that - this is recorded, not fixed.

## Verified

L1 (`variants/tests/l1_trainer.py`): `fixed_psnr`, `fixed_esrgan_corrected_shuffle`
(PSNR → ESRGAN with the pretrain proof), `fixed_esrgan_resume_3_5`,
`asconfigured_upstream_noshuffle_deadlr`, and the failure paths
`fail_pretrain_missing` (9), `fail_dead_lr` (8), `fail_no_tfrecord` (5),
`fail_nonfinite_fixed` (4): identical stdout and bit-identical checkpoints
against the old trainer.
