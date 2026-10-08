# DeepTextures + MatUp render loss - map montages and render sheets

Generated 2026-10-08 by `run_deeptex.ps1 -Stage eval-render`. Numbers behind the verdict: `render_summary.md` (render accuracy) and `perceptual_summary.md` (LPIPS / FLIP).

## Which checkpoint each row is

| tag | row in render sheets | row in map montages | checkpoint | step |
|---|---|---|---|---|
| `matsynth_all` | DeepTextures (matsynth_all) | DeepTextures (matsynth_all) | `/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k/ckpt-8` | 40000 |
| `render_r0a_30k` | DeepTextures grouped (R0a) 30k | - | `/workspace/runs/deeptex/render_r0a/checkpoints_all/esrgan_40k/ckpt-6` | 30000 |
| `render_r0a_35k` | DeepTextures grouped (R0a) 35k | - | `/workspace/runs/deeptex/render_r0a/checkpoints_all/esrgan_40k/ckpt-7` | 35000 |
| `render_r0a_40k` | DeepTextures grouped (R0a) 40k | DeepTextures (render_r0a) | `/workspace/runs/deeptex/render_r0a/checkpoints_all/esrgan_40k/ckpt-8` | 40000 |
| `render_r0b_30k` | DeepTextures grouped (R0b) 30k | - | `/workspace/runs/deeptex/render_r0b/checkpoints_all/esrgan_40k/ckpt-6` | 30000 |
| `render_r0b_35k` | DeepTextures grouped (R0b) 35k | - | `/workspace/runs/deeptex/render_r0b/checkpoints_all/esrgan_40k/ckpt-7` | 35000 |
| `render_r0b_40k` | DeepTextures grouped (R0b) 40k | DeepTextures (render_r0b) | `/workspace/runs/deeptex/render_r0b/checkpoints_all/esrgan_40k/ckpt-8` | 40000 |
| `render_r1_30k` | DeepTextures + render loss (R1) 30k | - | `/workspace/runs/deeptex/render_r1/checkpoints_all/esrgan_40k/ckpt-6` | 30000 |
| `render_r1_35k` | DeepTextures + render loss (R1) 35k | - | `/workspace/runs/deeptex/render_r1/checkpoints_all/esrgan_40k/ckpt-7` | 35000 |
| `render_r1_40k` | DeepTextures + render loss (R1) 40k | DeepTextures (render_r1) | `/workspace/runs/deeptex/render_r1/checkpoints_all/esrgan_40k/ckpt-8` | 40000 |

## Two different "render"s

- The `render` column of `<material>.png` and `overview_render.png` (deeptex_arms_report.py) is
  DeepTextures run as an IMAGE super-resolver on MatUp's LR render (`render_lr.png`), exactly as
  in every earlier montage. No predicted map goes into it.
- `render_<material>.png` and `overview_render_pred.png` (svbrdf_render_eval.py) are MatUp's own
  torch renderer applied to each method's PREDICTED SVBRDF - all five maps - under EVAL_LIGHT and
  held-out grazing / mid / high lights, plus an EVAL_LIGHT error heatmap at the fixed gain stated
  in the sheet header. That is the quantity the render loss trains, and the R1 verdict is read
  from its numbers (`render_summary.md`).

## Before reading

- The map montages show ONE checkpoint per arm (step 40000). This recipe's global colour swings
  from checkpoint to checkpoint (DEEPTEX-DEVIATIONS 0.10), so a single step cannot say whether
  an arm's colour is better. The render metrics cover every step in the table above, and
  `checkpoints/<arm>/` holds each arm's drift tables (`drift.md` baseColor, `drift_normal.md`) and its per-checkpoint `0844_syntetic_foam` sheets.
- Full numbers stay outside this folder: `runs/deeptex/render_eval/main/` (report.json, per_material.md,
  metrics_long.csv) and `runs/deeptex/report_render_loss/` (full-size sheets/).

## Files

| file | content |
|---|---|
| `<material>.png` | rows = LR / bilinear / MatUp / DeepTextures arms / GT, columns = five maps + render (image SR) |
| `overview_<map>.png` | rows = the evaluation materials, columns = methods |
| `summary.md`, `per_material.md`, `metrics_long.csv`, `report.json` | map-space metrics (deeptex_arms_report.py) |
| `render_<material>.png` | rows = methods, columns = renders of the predicted SVBRDF + EVAL_LIGHT error heatmap. Local only: ~13 MB each, so .gitignore keeps them out of git; eval-render regenerates them |
| `overview_render_pred.png` | rows = materials, columns = methods, EVAL_LIGHT |
| `render_summary.md` | render-space metrics (svbrdf_render_eval.py) |
| `perceptual_summary.md`, `perceptual_report.json` | LPIPS + FLIP on renders of the predicted SVBRDFs (svbrdf_perceptual_eval.py), written by `-Stage eval-perceptual`; its columns and steps are in its own header |
| `checkpoints/<arm>/` | colour drift per checkpoint (deeptex_drift.py) and its sheets |

## Regenerate

```powershell
powershell -ExecutionPolicy Bypass -File scripts/deeptex/run_deeptex.ps1 -Stage eval-render -Arms render_r0a,render_r0b,render_r1 -Steps 30000,35000,40000
# then, on the same folders (perceptual_summary.md, perceptual_report.json):
powershell -ExecutionPolicy Bypass -File scripts/deeptex/run_deeptex.ps1 -Stage eval-perceptual -Arms render_r0a,render_r0b,render_r1 -Steps 30000,35000,40000
```
