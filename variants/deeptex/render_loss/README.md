# render_loss/ - RL: MatUp's render loss in DeepTextures training

R0a / R0b / R1 of `runs/montages/deeptex_render_loss/` and
`docs/RENDER-LOSS-ABLATION.md`. A declared experiment outside the paper.

    python variants/deeptex/paper_loop/main.py --experiment render --cfg_path /workspace/variants/deeptex/render_loss/r1.yaml \
        --stage esrgan --losses upstream --shuffle off --allow-dead-lr-schedule --pretrain-dir none \
        --no-val --log-every 100 --render-monitor-every 100

(the argv run_deeptex.ps1 -Stage render-train always passed, unchanged).

## What this arm changes against the paper

Exactly `hooks.py` plus `../hooks_common/` (the old trainer's machinery shared
with the fixed arm). Hook by hook, with the paper line each replaces:

| hook | `train_esrgan.py` | here |
|---|---|---|
| `load_cfg` / `experiment_cfg` | `:28` | grouped config required (exit 8 otherwise), `--stage esrgan` only (exit 8), `ch_size` 3, **seed required**, `w_render` resolved to a Python float (traced as a constant), render block validated, out roots checked; prints `[*] grouped mode: ...` |
| `pre_build` | - | `check_gpu` (first random draw), grouped record check (schema + verify_report sha256), disk |
| `build_dataset` | `:37` `shuffle=False` | no dataset here; the grouped stream is built after the restore (`resume_dataset`) at `start_step * batch`, so a resumed run continues the exact crop sequence |
| `build_losses` | `:52-53` | upstream RaGAN unchanged, then `RenderLoss` (after the seeded init; it draws no random numbers) |
| `restore` | `:65-81` | `hooks_common/ckpt.py` (verified resume; `--pretrain-dir` proven, used by the overfit gate), then `[init] step= g_sha= d_sha=` |
| `step_function` | `:84` | `tf.function(input_signature=...)`: one trace for the run |
| `pre_tape` | in `:85` | (B,5,s,s,3) → (5B,s,s,3) row-major; GT render target computed before the tape |
| `extra_loss` | after `:98` | `losses_G['render'] = w_render * terms.render_term(...)`, added after `gan`, so `tf.add_n` sums reg, pixel, feature, gan, render |
| `pre_apply` | `:106-109` | `assert_all_finite(global_norm(grads_G + grads_D))` as a control dependency of both updates |
| `step_outputs` | `:111` | render diagnostics: `mask_frac`, `oor_*`, `sr_nz_guard` |
| `call_step` | `:123` | `[render] step= l21=` probe (step 1, every N, last) before the update; non-finite loss or gradient → `nan_debug_exit` (state + batch to `<sub_name>_nan_debug`, exit 4; the arm's own checkpoints are never touched) |
| `extra_report` | `:125-128` | `[terms] step= reg= pixel= feature= gan= render= mask_frac= oor_* sr_nz_guard=` |

Not in this arm (the fixed arm's repairs): corrected RaGAN (#8), shuffle (#5),
PSNR pretrain (#2 as a recipe). `--losses`, `--shuffle` accept only `upstream`
/ `off`. The dead-LR check (#3) and the C++ log level (#4) come with
hooks_common, as in the old trainer: the render arm prints the
`--allow-dead-lr-schedule` warning exactly as before.

## Kept as it was, and named

* **max_to_keep is 3.** The eight checkpoints of a 40k run are the
  `-KeepAllCheckpoints` copier's (run.ps1 → `variants/util/lib.ps1`), not the
  trainer's.
* `check_grouped_dataset`'s message still names
  `scripts\deeptex\run_deeptex.ps1 -Stage prep-grouped` - kept byte-identical
  with the old trainer; the new entry is `variants\deeptex\render_loss\run.ps1`.

## Files

| file | from | how |
|---|---|---|
| `hooks.py` | `scripts/deeptex/deeptex_train.py` (grouped branch) | decomposed; L0 `moves/deeptex_trainer.json`, L1 |
| `grouped_dataset.py`, `render_tf.py` | `scripts/deeptex/` | moved (AST-identical, L0) |
| `terms.py` | `deeptex_train.py:741-743`, `render_calibrate.py:344-369` | the one place the render term (and calibrate's G terms) are built |
| `gates/render_gate.py`, `gates/render_parity_torch.py` | `scripts/deeptex/`, `scripts/matup/` | moved; L4 |
| `gates/render_calibrate.py` | `scripts/deeptex/render_calibrate.py` | decomposed onto `terms.py`; L4 (w* = 10, unrounded 9.54) |
| `r0a.yaml`, `r0b.yaml`, `r1.yaml` | `configs/deeptex/esrgan_matsynth_render_*.yaml` | copied byte for byte (top-level keys are read by the ps1 line regexes) |

## Depends on (by path only)

* `runs/deeptex/datasets/matsynth/` pairs and `.../matsynth_grouped/` record
  (prep-matsynth, prep-grouped);
* the overfit gate and calibration read the **matsynth_all** (and chratio)
  checkpoints `runs/deeptex/matsynth_{all,chratio}/checkpoints/esrgan_40k`;
* every DT evaluation renders against MatUp's `render_lr.png` / `render_gt.png`
  in `runs/matsynth/<material>/` (matup/baseline).

## run.ps1

Stages `prep-grouped`, `gate-render` (`-Gate parity|calibrate|overfit|smoke|premise`),
`render-train` (`-Arm r0a|r0b|r1`), `eval-render`, `eval-perceptual`, plus
`preflight`; `-Stage` is mandatory. Generated from `scripts/deeptex/run_deeptex.ps1`: each stage function below is the old one verbatim except the script/config paths (old line ranges in the file header). Shared helpers come from `variants/util/lib.ps1`. L5 (`variants/tests/l5_deeptex.ps1`) runs every stage through both and compares the docker argv, the console transcript and the files created.
The trainer calls are `paper_loop/main.py --experiment render` + the old flags;
every `deeptex_compare --maps all --scalar-collapse mean` is `matsynth/eval_arms.py`.
The README `eval-render` writes into `runs/montages/deeptex_render_loss/` now names
`variants/deeptex/render_loss/run.ps1` in its regenerate commands.
