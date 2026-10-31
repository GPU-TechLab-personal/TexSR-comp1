# DeepTextures arms only - maps and renders of the predicted SVBRDF

Generated 2026-10-08 by `scripts/compare/deeptex_dt_montage.py`. A DeepTextures-only view of the render-loss study in `..`: every row is a DeepTextures checkpoint or GT. There are NO LR / bilinear / MatUp rows here - those baselines are in `../<material>.png` and `../render_<material>.png`.

## Rows (top to bottom, the `--arm` order, GT last)

| # | row label | tag | arm folder | checkpoint prefix (sr_manifest.json) | step |
|---|---|---|---|---|---|
| 1 | DeepTextures (matsynth_all) 40k | `matsynth_all` | `/workspace/runs/deeptex/eval/matsynth_all` | `/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k/ckpt-8` | 40000 |
| 2 | DeepTextures grouped (R0a) 30k | `render_r0a_30k` | `/workspace/runs/deeptex/eval/render_r0a_30k` | `/workspace/runs/deeptex/render_r0a/checkpoints_all/esrgan_40k/ckpt-6` | 30000 |
| 3 | DeepTextures grouped (R0a) 35k | `render_r0a_35k` | `/workspace/runs/deeptex/eval/render_r0a_35k` | `/workspace/runs/deeptex/render_r0a/checkpoints_all/esrgan_40k/ckpt-7` | 35000 |
| 4 | DeepTextures grouped (R0a) 40k | `render_r0a_40k` | `/workspace/runs/deeptex/eval/render_r0a_40k` | `/workspace/runs/deeptex/render_r0a/checkpoints_all/esrgan_40k/ckpt-8` | 40000 |
| 5 | DeepTextures grouped (R0b) 30k | `render_r0b_30k` | `/workspace/runs/deeptex/eval/render_r0b_30k` | `/workspace/runs/deeptex/render_r0b/checkpoints_all/esrgan_40k/ckpt-6` | 30000 |
| 6 | DeepTextures grouped (R0b) 35k | `render_r0b_35k` | `/workspace/runs/deeptex/eval/render_r0b_35k` | `/workspace/runs/deeptex/render_r0b/checkpoints_all/esrgan_40k/ckpt-7` | 35000 |
| 7 | DeepTextures grouped (R0b) 40k | `render_r0b_40k` | `/workspace/runs/deeptex/eval/render_r0b_40k` | `/workspace/runs/deeptex/render_r0b/checkpoints_all/esrgan_40k/ckpt-8` | 40000 |
| 8 | DeepTextures + render loss (R1) 30k | `render_r1_30k` | `/workspace/runs/deeptex/eval/render_r1_30k` | `/workspace/runs/deeptex/render_r1/checkpoints_all/esrgan_40k/ckpt-6` | 30000 |
| 9 | DeepTextures + render loss (R1) 35k | `render_r1_35k` | `/workspace/runs/deeptex/eval/render_r1_35k` | `/workspace/runs/deeptex/render_r1/checkpoints_all/esrgan_40k/ckpt-7` | 35000 |
| 10 | DeepTextures + render loss (R1) 40k | `render_r1_40k` | `/workspace/runs/deeptex/eval/render_r1_40k` | `/workspace/runs/deeptex/render_r1/checkpoints_all/esrgan_40k/ckpt-8` | 40000 |
| 11 | GT | - | `/workspace/runs/matsynth/<material>/GT` | - | - |

Each row is ONE checkpoint. This recipe's global colour swings from checkpoint to checkpoint (DEEPTEX-DEVIATIONS 0.10), so judge an arm across its 30k / 35k / 40k rows, not from one cell. Thin grey lines separate the arm groups and GT.

## Files

| file | rows | columns | MB |
|---|---|---|---|
| `dt_0559_brick_uneven_stones.png` | the arms + GT | 8 (below) | 6.6 |
| `dt_0844_syntetic_foam.png` | the arms + GT | 8 (below) | 6.7 |
| `dt_1587_cobblestones.png` | the arms + GT | 8 (below) | 6.7 |
| `dt_2119_corten_steel.png` | the arms + GT | 8 (below) | 7.5 |
| `dt_acg_concrete_026.png` | the arms + GT | 8 (below) | 8.1 |
| `dt_acg_metal_026.png` | the arms + GT | 8 (below) | 7.3 |
| `dt_acg_plastic_009.png` | the arms + GT | 8 (below) | 3.9 |
| `dt_acg_roofing_tiles_006.png` | the arms + GT | 8 (below) | 5.2 |
| `dt_acg_wood_floor_060.png` | the arms + GT | 8 (below) | 5.7 |
| `dt_tc_metal_029.png` | the arms + GT | 8 (below) | 6.7 |
| `dt_th_brick_wall_005.png` | the arms + GT | 8 (below) | 7.2 |
| `dt_th_brown_mud_leaves_001.png` | the arms + GT | 8 (below) | 8.3 |
| `dt_overview_baseColor.png` | materials | the arms + GT, the stored `baseColor` map | 5.4 |
| `dt_overview_normal.png` | materials | the arms + GT, the stored `normal` map | 4.8 |
| `dt_overview_roughness.png` | materials | the arms + GT, the stored `roughness` map | 2.2 |
| `dt_overview_metallic.png` | materials | the arms + GT, the stored `metallic` map | 0.9 |
| `dt_overview_height.png` | materials | the arms + GT, the stored `height` map | 2.3 |
| `dt_overview_render.png` | materials | the arms + GT, render under EVAL_LIGHT of the predicted SVBRDF | 5.7 |

## Columns of `dt_<material>.png`

| column | content |
|---|---|
| baseColor, normal, roughness, metallic, height | the stored PNG of the row: the arm's `<arm folder>/<material>/<map>.png` (8-bit), GT's `GT/<map>.png` (8-bit; height 16-bit) |
| render EVAL_LIGHT | MatUp's torch renderer on the row's PREDICTED SVBRDF (all five maps), EVAL_LIGHT [0.276, 0.276, 0.921], norm 1, 67 deg elevation |
| render held-out grazing | the same, under held-out light #4 (13 deg elevation): the 'grazing' sheet light of svbrdf_render_eval (1.4 x fib(16) turned 43.25 deg about z, >= 2.5 deg from every training light direction) |
| \|pred - GT\| x4 (EVAL) | mean over RGB of \|render - GT render\| under EVAL_LIGHT, times HEAT_GAIN = 4 (FIXED for every row and material; >= 0.25 saturates), inferno colour map; grey = texels whose GT normal fails training's validity rule. Blank for the GT row |

## How the cells are made

- **Crop.** Per material, the border is what `matup_postprocess.detect_border` measures on MatUp's `_opt` maps of the run named in the sheet title (baseColor / roughness / metallic; never normal or the pre-cropped height) - the call `deeptex_arms_report.py` and `svbrdf_render_eval.py` make. Map cells and renders cover the same 496^2 px region.
- **Maps** are drawn as `deeptex_arms_report.py` draws them: `post.deeptex_column` / the GT crop, `matup_montage.to_display` (16-bit >> 8, area-downsampled). Nothing is normalised: an all-zero metallic is black, the normal is its stored OpenGL RGB encoding, grey maps are grey.
- **Renders** follow `svbrdf_render_eval.py` exactly: `load_dt` / `load_gt` (`matup_render.load_set(..., opengl=True, floor roughness)` + `position3d` of the set's own height), `prepare` (crop, `get_renorm_clamped_maps`), MatUp's renderer UNGUARDED, `to_cell` / `heat_cell`. The cells equal those of `../render_<material>.png` at a smaller size. The GT row's EVAL_LIGHT render reproduces MatUp's committed `render_gt.png` (max difference per material below, tolerance 2.0 LSB).

## Two meanings of "render"

- HERE (and in `../render_<material>.png`, `../overview_render_pred.png`): the PREDICTED SVBRDF - all five maps a checkpoint produced - rendered by MatUp's renderer. This is the quantity the render loss trains.
- In `../<material>.png` and `../overview_render.png` (deeptex_arms_report.py): DeepTextures run as an IMAGE super-resolver on MatUp's LR render (`render_lr.png`). No predicted map goes into it. That column is NOT in this folder.

## Per material

| material | MatUp run (border source) | border | crop | GT-valid texels | GT self-check (LSB) |
|---|---|---|---|---|---|
| 0559_brick_uneven_stones | Sep09_11h48m34s | 8 | 496^2 | 1.0000 | 1.00 |
| 0844_syntetic_foam | Sep09_11h49m44s | 8 | 496^2 | 0.9996 | 1.00 |
| 1587_cobblestones | Sep09_11h50m54s | 8 | 496^2 | 1.0000 | 1.00 |
| 2119_corten_steel | Sep09_11h52m04s | 8 | 496^2 | 1.0000 | 1.00 |
| acg_concrete_026 | Sep09_11h53m13s | 8 | 496^2 | 1.0000 | 1.00 |
| acg_metal_026 | Sep09_11h54m23s | 8 | 496^2 | 1.0000 | 1.00 |
| acg_plastic_009 | Sep09_11h55m31s | 8 | 496^2 | 1.0000 | 1.00 |
| acg_roofing_tiles_006 | Sep09_11h56m44s | 8 | 496^2 | 1.0000 | 1.00 |
| acg_wood_floor_060 | Sep09_11h57m57s | 8 | 496^2 | 1.0000 | 1.00 |
| tc_metal_029 | Sep09_11h59m07s | 8 | 496^2 | 1.0000 | 1.00 |
| th_brick_wall_005 | Sep09_12h00m21s | 8 | 496^2 | 1.0000 | 1.00 |
| th_brown_mud_leaves_001 | Sep09_12h01m35s | 8 | 496^2 | 1.0000 | 1.00 |

Missing arm outputs (drawn as n/a): none.
Texels whose rendered normal had n_z == 0 exactly (given the smallest positive n_z, as svbrdf_render_eval does): none.

## Regenerate

From the repo root (the GPU is used for the renders; a few minutes):

```
docker compose -f docker/compose.yaml run --rm matup python /workspace/scripts/compare/deeptex_dt_montage.py --arm matsynth_all=/workspace/runs/deeptex/eval/matsynth_all --arm render_r0a_30k=/workspace/runs/deeptex/eval/render_r0a_30k --arm render_r0a_35k=/workspace/runs/deeptex/eval/render_r0a_35k --arm render_r0a_40k=/workspace/runs/deeptex/eval/render_r0a_40k --arm render_r0b_30k=/workspace/runs/deeptex/eval/render_r0b_30k --arm render_r0b_35k=/workspace/runs/deeptex/eval/render_r0b_35k --arm render_r0b_40k=/workspace/runs/deeptex/eval/render_r0b_40k --arm render_r1_30k=/workspace/runs/deeptex/eval/render_r1_30k --arm render_r1_35k=/workspace/runs/deeptex/eval/render_r1_35k --arm render_r1_40k=/workspace/runs/deeptex/eval/render_r1_40k --out /workspace/runs/montages/deeptex_render_loss/dt_only
```
