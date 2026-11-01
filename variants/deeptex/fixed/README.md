# variants/deeptex/fixed/

FX, the fixed arm of `docs/DEEPTEX-DEVIATIONS.md`: the DeepTextures paper loop
with its code defects repaired. It trains a PSNR pre-train stage on
`textures_db`, then ESRGAN from it.

## Contents

| file | what it does |
|---|---|
| `hooks.py` | the arm's hook bundle on top of `hooks_common/trainer.py`: its flags (`--stage`, `--losses`, `--shuffle`), refuses grouped / render configs, shuffled dataset, corrected or upstream RaGAN; `build(argv)` returns the hooks and `psnr_loop` or `loop` |
| `losses_corrected.py` | RaGAN generator / discriminator losses with `from_logits=True` on the raw logits (upstream applies a sigmoid, then `from_logits=False`) |
| `psnr_texsr.yaml` | stage 1: PSNR pre-train (G only, pixel loss), batch 16, niter 40000, lr_steps scaled to [8000, 16000, 24000, 32000]; `sub_name: psnr_texsr` |
| `esrgan_scaled.yaml` | stage 2: ESRGAN, the config the comparison runs on; lr_steps scaled into the run ([5000, 10000, 20000, 30000]), `pretrain_dir: /workspace/runs/deeptex/checkpoints/psnr_texsr`; `sub_name: esrgan_texsr` |
| `esrgan_asconfigured.yaml` | upstream's recipe through this trainer, both defects left in (`pretrain_dir: null`, lr_steps that never fire at niter 40000); needs `--losses upstream --shuffle off --allow-dead-lr-schedule` |
| `run.ps1` | orchestrator: stages `preflight`, `pretrain`, `train` |

## How it works

`run.ps1` runs `paper_loop/main.py --experiment fixed` in the deeptex service of
`docker/compose.yaml`:

| stage | runs |
|---|---|
| `preflight` | `Invoke-DeeptexPreflight` (`variants/util/lib.ps1`): stray containers, >= 6000 MiB free VRAM, >= 15 GB free disk. Runs before every stage unless `-SkipPreflight` |
| `pretrain` | `--stage psnr` with `-PsnrConfig` -> `paper_loop/psnr_loop.py` (the donor's `train_psnr.py`) |
| `train` | `--stage esrgan` with `-Config` -> `paper_loop/loop.py` (`train_esrgan.py`) |

Both training stages pass `--tfrecord /workspace/runs/deeptex/datasets/<Dataset>/<Dataset>_lrx4.tfrecord`
and validate on `.../<Dataset>/HR_val`, `.../<Dataset>/LRx4_val` (`-Dataset`
defaults to `textures_db`; the record is built by `variants/deeptex/upstream_runner/`).
Checkpoints go to `<checkpoint_root>/<sub_name>`, logs to `<log_root>/<sub_name>`
(`/workspace/runs/deeptex/{checkpoints,logs}` in all three configs), so the
ESRGAN stage finds the PSNR checkpoint at `runs/deeptex/checkpoints/psnr_texsr`.
`run.ps1` reads `sub_name` and `niter` from the config with line regexes, maps
the trainer's output to a failure kind (`Get-FailureKind`) and checks that a
`[metrics] step=<niter>` line was reached.

### What this arm changes against the paper

| # | defect (DEEPTEX-DEVIATIONS) | upstream | here |
|---|---|---|---|
| #2 | pretrain lookup | `train_esrgan.py:71` `pretrain_dir = cfg['pretrain_name']` = `'./checkpoints/'`, misses `./checkpoints/psnr_pretrain/`, prints and trains from random init | `pretrain_dir` (absolute), exit 9 when nothing is there, restore **proven** by a weight delta (`hooks_common/ckpt.py`) |
| #3 | dead LR schedule | `lr_steps[0] = 50000 >= niter = 40000` | refused (exit 8) unless `--allow-dead-lr-schedule` (`hooks_common/safety.check_lr_schedule`); `esrgan_scaled.yaml` has a live one |
| #4 | log levels | `:20` C++ level set after the import; `:23-25` Python logger silenced | C++ level 2 before the import, Python logger at WARNING (`paper_loop/main.py`, `hooks_common/trainer.on_start`) |
| #5 | frozen batch order | `:37` `shuffle=False` | `shuffle=True`; `--shuffle off` restores (`hooks.py build_dataset`) |
| #8 | RaGAN on sigmoids | `modules/losses.py:56,80` `from_logits=False` after a manual sigmoid | `losses_corrected.py`, `from_logits=True`; `--losses upstream` restores (`hooks.py build_losses`) |
| - | restore unchecked | `:66,:73` status discarded | `assert_existing_objects_matched()` on resume |
| - | no numerics check | - | non-finite loss -> `[fatal] non-finite loss`, save, exit 4 |
| - | no validation | - | `--val-every` (default 2000) on the held-out split (`hooks_common/val.py`) |

## Run

    variants\deeptex\fixed\run.ps1 -Stage pretrain
    variants\deeptex\fixed\run.ps1 -Stage train -Config variants/deeptex/fixed/esrgan_scaled.yaml

`-MaxSteps N` truncates a run (smoke test); an existing checkpoint directory is
resumed unless `-Force -DeleteCheckpoints` moves it aside. The trainer call
`run.ps1` makes:

    python /workspace/variants/deeptex/paper_loop/main.py --experiment fixed --cfg_path /workspace/<cfg> \
        --stage psnr|esrgan --tfrecord <TFREC> --val-hr <DS>/HR_val --val-lr <DS>/LRx4_val [--max-steps N]

Measured VRAM peaks on this card (200-step probes, recorded in
`Invoke-DeeptexPreflight`): 4,491 MiB for the PSNR stage (batch 16), 3,234 MiB
for ESRGAN (batch 8).

## Known behaviour and defects

* **`-Config` defaults to `variants/deeptex/paper/esrgan_paper.yaml`**, so
  `-Stage train` without `-Config` runs this trainer on upstream's config (and
  stops at the dead-LR check, exit 8). Pass
  `-Config variants/deeptex/fixed/esrgan_scaled.yaml` for the fixed recipe.
* No stage runs `esrgan_asconfigured.yaml`; only `-Config` reaches it, and the
  stage does not add the three flags it needs.
* The `param` block is the one shared by every deeptex `run.ps1`: `-Recipe`,
  `-Arm`, `-Gate`, `-Arms`, `-Steps`, `-EvalDevice` and the other render /
  MatSynth parameters are accepted and ignored here.
* A grouped config (`train_dataset.grouped: True`) or one with `w_render` /
  `render:` is refused (exit 8): it belongs to `--experiment render`.
