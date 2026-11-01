# variants/deeptex/render_loss/

RL: MatUp's render loss inside DeepTextures training, a declared experiment
outside the paper. Arms R0a / R0b (w_render = 0, seeds 1 / 2) and R1 (seed 1,
calibrated w_render) ask whether a render-space term improves the predicted
SVBRDFs; results in `runs/montages/deeptex_render_loss/`, design in
`docs/RENDER-LOSS-ABLATION.md`.

## Contents

| file | what it does |
|---|---|
| `hooks.py` | `RenderHooks`, the arm's hook bundle on `hooks_common/trainer.py`: grouped batches, render term, `[render]` / `[terms]` / `[init]` lines, NaN debug exit; its flags (`--w-render`, `--overfit-batches`, `--overfit-fixed`, `--render-monitor-every`) |
| `grouped_dataset.py` | grouped MatSynth loader: one Example = one material's five maps (HR + LR); one crop + flip/rot per material, stateless randomness keyed on (seed, sample index), resume at `start_index`, plus the MatUp position grid `pos` |
| `render_tf.py` | MatUp's SVBRDF renderer ported line for line to TensorFlow; `decode_maps` (one decode for SR, GT and gates, incl. the GT validity mask), `RenderLoss` (l2,1 over 100 training lights, `light_chunks`), `validate_render_cfg` |
| `terms.py` | the render term (`render_term`), one weighted G term as the step builds it (`g_term`, calibrate only), and `remove_offset` |
| `r0a.yaml`, `r0b.yaml`, `r1.yaml` | the three arms: grouped record, batch 2 materials (x 5 maps = 10 images), niter 40000, seed 1 / 2 / 1, `w_render` 0.0 / 0.0 / 10.0, `render:` block, `checkpoint_root` `/workspace/runs/deeptex/render_<arm>/checkpoints` |
| `run.ps1` | orchestrator: `preflight`, `prep-grouped`, `gate-render`, `render-train`, `eval-render`, `eval-perceptual` |
| `build/` | builds and verifies the grouped TFRecord (own README) |
| `gates/` | the pre-training gates, below |
| `eval/` | render-space evaluation of the predicted SVBRDFs (own README) |

`gates/`:

| file | container | what it does |
|---|---|---|
| `gates/render_gate.py` | deeptex, GPU | `--part export` writes `parity_tf.npz` for the torch side; `grads`, `flow`, `grid`, `chunks`, `mask`, `graph` check the TF render loss (finite gradients, gradient reaches the maps, flip/rot equivariance, `light_chunks` 4 == 1, mask arithmetic, no MatMul/Einsum ops) |
| `gates/render_parity_torch.py` | matup | MatUp's own torch renderer against the TF port: light table bit for bit (`parity_lights.json`), core / decode / anchored checks against a float64 oracle (`parity_torch.json`) |
| `gates/render_calibrate.py` | deeptex, GPU | chooses `w_render`: median over six trained states of max(|g_pixel|, |g_feature|, |g_gan|) / |g_render_offrm|, rounded to 1 significant figure; writes `calibration.json`, never edits a config |

Each gate writes `<part>.json` with `pass` into `/workspace/runs/deeptex/render_gate/`
and deletes its stale JSON first.

## How it works

Training: `paper_loop/main.py --experiment render` builds `RenderHooks` and runs
`paper_loop/loop.py`. Against the paper, the changes are exactly `hooks.py` plus
`../hooks_common/`:

| hook | `train_esrgan.py` | here |
|---|---|---|
| `load_cfg` / `experiment_cfg` | `:28` | grouped config required, `--stage esrgan` only, `ch_size` 3, **seed required**, `w_render` resolved to a Python float (traced as a constant), render block validated, out roots checked (all exit 8); prints `[*] grouped mode: ...` |
| `pre_build` | - | `check_gpu` (first random draw), grouped record check (schema + `verify_report.json` sha256), disk |
| `build_dataset` | `:37` `shuffle=False` | no dataset here; the grouped stream is built after the restore (`resume_dataset`) at `start_step * batch`, so a resumed run continues the exact crop sequence |
| `build_losses` | `:52-53` | upstream RaGAN unchanged, then `RenderLoss` (after the seeded init; it draws no random numbers) |
| `restore` | `:65-81` | `hooks_common/ckpt.py` (checked resume; `--pretrain-dir` proven, used by the overfit gate), then `[init] step= g_sha= d_sha=` |
| `step_function` | `:84` | `tf.function(input_signature=...)`: one trace for the run |
| `pre_tape` | in `:85` | (B,5,s,s,3) -> (5B,s,s,3) row-major; GT render target computed before the tape |
| `extra_loss` | after `:98` | `losses_G['render'] = w_render * terms.render_term(...)`, added after `gan`, so `tf.add_n` sums reg, pixel, feature, gan, render |
| `pre_apply` | `:106-109` | `assert_all_finite(global_norm(grads_G + grads_D))` as a control dependency of both updates |
| `step_outputs` | `:111` | render diagnostics: `mask_frac`, `oor_*`, `sr_nz_guard` |
| `call_step` | `:123` | `[render] step= l21=` probe (step 1, every `--render-monitor-every`, last) before the update; non-finite loss or gradient -> `nan_debug_exit` (state + batch to `<sub_name>_nan_debug`, exit 4; the arm's own checkpoints are never touched) |
| `extra_report` | `:125-128` | `[terms] step= reg= pixel= feature= gan= render= mask_frac= oor_* sr_nz_guard=` |
| `write_summaries` | in `:131` | `diag/*` scalars |

Not in this arm (the fixed arm's repairs): corrected RaGAN (#8), shuffle (#5),
PSNR pretrain (#2 as a recipe). The dead-LR check (#3) and the C++ log level
(#4) come with hooks_common, so the arm prints the `--allow-dead-lr-schedule`
warning.

Pipeline (`run.ps1`), with its inputs and outputs:

| stage | what it runs | reads | writes |
|---|---|---|---|
| `prep-grouped` | `build/texsr_build_grouped_tfrecord.py` (skipped if the record exists, unless `-Force`), then `build/texsr_verify_grouped.py` | `runs/deeptex/datasets/matsynth/` (from `matsynth/run.ps1 -Stage prep-matsynth`) | `runs/deeptex/datasets/matsynth_grouped/` |
| `gate-render -Gate parity` | `render_gate.py --part export` (deeptex) -> `render_parity_torch.py --part all` (matup) -> `render_gate.py --part grads,flow,grid,chunks,mask,graph` (deeptex); a failed part does not stop the others | `r1.yaml` render block, `runs/matsynth/<m>/`, MatSynth pairs | `runs/deeptex/render_gate/{export,parity_lights,parity_torch,grads,flow,grid,chunks,mask,graph}.json`, `parity_tf.npz` |
| `gate-render -Gate calibrate` | `render_calibrate.py` on matsynth_all + matsynth_chratio at 30k/35k/40k (resolved by step tensor), 20 batches, seed 0 | `runs/deeptex/matsynth_{all,chratio}/checkpoints[_all]/esrgan_40k` | `runs/deeptex/render_gate/calibration.json`; prints w* to write into `r1.yaml` by hand |
| `gate-render -Gate overfit` | R0a (w = 0) and R1 (w*) from matsynth_all 40k on one fixed batch, 300 steps; pass: R1(300) <= 0.6 R1(1) and R1(300) <= 0.85 R0a(300) | `runs/deeptex/matsynth_all/checkpoints/esrgan_40k`, `calibration.json` | `runs/deeptex/render_gate/overfit.json`, runs under `runs/deeptex/_smoke_render/overfit_*` |
| `gate-render -Gate smoke` | R0a and R1: 100 steps from scratch, resume to 200; same `[init]` sha for both; measures steps/s and VRAM peak; then `matsynth/eval_arms.py` on the R1 smoke checkpoint for `acg_concrete_026` (6 PNGs expected) | `calibration.json` | `runs/deeptex/render_gate/smoke.json`, `runs/deeptex/_smoke_render/` |
| `gate-render -Gate premise` | `eval_arms.py` + `eval/svbrdf_render_eval.py` on matsynth_all + matsynth_chratio at 30k/35k/40k | their checkpoints | `runs/deeptex/eval/`, `runs/deeptex/render_eval/premise/` |
| `render-train -Arm r0a\|r0b\|r1` | the trainer (argv below); refuses `w_render: null`, and for r1 a config value that differs from `calibration.json`'s w*; VRAM check against `smoke.json`'s peak + 1500 MiB; `-KeepAllCheckpoints` copies every checkpoint to `checkpoints_all` | `r<arm>.yaml`, the grouped record + `verify_report.json` | `runs/deeptex/render_<arm>/{checkpoints,checkpoints_all,logs}/esrgan_40k` |
| `eval-render` | per (arm, step) `matsynth/eval_arms.py` -> `eval/svbrdf_render_eval.py` (matup) -> `matsynth/deeptex_arms_report.py` at the last step (dataprep) -> `matsynth/deeptex_drift.py` per arm for baseColor and normal -> collect + write the folder's README | `runs/deeptex/eval/matsynth_all` (complete at 40k), arm checkpoints, `runs/matsynth/<m>/` | `runs/deeptex/{eval,render_eval/main,report_render_loss,drift}/`, collected into `runs/montages/deeptex_render_loss/` |
| `eval-perceptual` | `eval/svbrdf_perceptual_eval.py` (exp service, `docker/exp/compose.yaml`) on the eval folders `eval-render` scored; super-resolves nothing | `runs/deeptex/eval/<arm>_<k>k` | `runs/deeptex/render_eval/perceptual/`; `perceptual_summary.md`, `perceptual_report.json` copied to `runs/montages/deeptex_render_loss/` |

`smoke_*` arm tags in `eval-render` / `eval-perceptual` keep every output under
`runs/deeptex/_smoke_render/`. Other inputs, by path only: every DT evaluation
renders against MatUp's `render_lr.png` / `render_gt.png` in
`runs/matsynth/<material>/` (`variants/matup/baseline/`).

## Run

    variants\deeptex\render_loss\run.ps1 -Stage prep-grouped
    variants\deeptex\render_loss\run.ps1 -Stage gate-render -Gate parity
    variants\deeptex\render_loss\run.ps1 -Stage gate-render -Gate calibrate
    variants\deeptex\render_loss\run.ps1 -Stage gate-render -Gate overfit
    variants\deeptex\render_loss\run.ps1 -Stage gate-render -Gate smoke
    variants\deeptex\render_loss\run.ps1 -Stage gate-render -Gate premise
    variants\deeptex\render_loss\run.ps1 -Stage render-train -Arm r1 -KeepAllCheckpoints
    variants\deeptex\render_loss\run.ps1 -Stage eval-render -Arms render_r0a,render_r0b,render_r1 -Steps 30000,35000,40000
    variants\deeptex\render_loss\run.ps1 -Stage eval-perceptual -Arms render_r0a,render_r0b,render_r1 -Steps 30000,35000,40000

`-Stage render-train` calls the trainer with

    python /workspace/variants/deeptex/paper_loop/main.py --experiment render --cfg_path /workspace/variants/deeptex/render_loss/r1.yaml \
        --stage esrgan --losses upstream --shuffle off --allow-dead-lr-schedule --pretrain-dir none \
        --no-val --log-every 100 --render-monitor-every 100

An existing checkpoint directory is resumed (run.ps1 then warns that the
R0a/R1 pairing no longer holds); `-Force -DeleteCheckpoints` moves the arm's
`checkpoints` and `checkpoints_all` aside. `-EvalDevice cpu` (with
`-EvalThreads`, default 4) runs `eval-perceptual` beside a training job.

Measured (`runs/deeptex/render_gate/`): calibration w* = 10 (unrounded 9.54,
spread 1.49x over the six states); overfit R1 l21 0.244 -> 0.105 in 300 steps
against R0a 0.195; smoke R0a 3.96 steps/s, peak 3,882 MiB, ~2.8 h per 40k steps,
R1 3.66 steps/s, peak 4,919 MiB, ~3.0 h per 40k steps. Over 7,500 MiB on R1,
set `render.light_chunks: 4` (same loss, lights in recomputed chunks).

## Known behaviour and defects

* **max_to_keep is 3.** The eight checkpoints of a 40k run are the
  `-KeepAllCheckpoints` copier's (`variants/util/lib.ps1`), not the trainer's.
  Without it only the last three survive.
* **`w_render` is written by hand.** `-Gate calibrate` prints w*; it goes into
  `r1.yaml` as `w_render: !!float <w>` (the tag matters: PyYAML reads an
  untagged `3e-2` as a string). Overfit and smoke read w* from
  `calibration.json` and pass it as `--w-render`.
* `--losses`, `--shuffle` accept only `upstream` / `off`, the values this arm
  trains with; `--stage psnr` exits 8.
* `check_grouped_dataset`'s `[fatal]` message for a missing record names
  `scripts\deeptex\run_deeptex.ps1 -Stage prep-grouped`, a deleted script; the
  stage is `variants\deeptex\render_loss\run.ps1 -Stage prep-grouped`. Comments
  in `r*.yaml` and docstrings in this folder name deleted `scripts/deeptex/`
  and `configs/deeptex/` paths the same way (`calibration.json` records
  `scripts/deeptex/render_calibrate.py`, the script that wrote it).
* The `param` block is the one shared by every deeptex `run.ps1`; parameters
  of other arms (`-Recipe`, `-PsnrConfig`, `-PaperRoot`, ...) are accepted and
  ignored.
