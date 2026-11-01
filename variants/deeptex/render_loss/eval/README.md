# variants/deeptex/render_loss/eval/

Judges the render-loss arms in render space. Every DeepTextures arm's
*predicted* SVBRDF (the five maps) is decoded as MatUp decodes a MatSynth folder
and rendered by MatUp's own torch renderer, then scored against MatUp's GT
render. This is the question R0a / R0b / R1 ask; the "render" column of
`deeptex_arms_report.py` (DeepTextures super-resolving MatUp's LR render) does
not answer it.

## Contents

| file | container | what it does |
|---|---|---|
| `svbrdf_render_eval.py` | matup | renders bilinear, MatUp and every `--arm` column under training, held-out and eval lights; raw and oracle-offset-removed (`off`) variants; render PSNR / l2,1 (P1 held-out, P2 train100), per-band l2,1, per-map swap l2,1, normal angle, render `|grad|` vs GT; self-check against the committed renders (exit 3) |
| `svbrdf_perceptual_eval.py` | exp (`docker/exp/compose.yaml`) | LPIPS and FLIP on the same renders, imported from `svbrdf_render_eval` (columns, maps, lights, self-check): whether R1's lost render detail was noise or detail GT has |
| `deeptex_dt_montage.py` | matup | DeepTextures-only contact sheets: the arms against each other and GT, no LR / bilinear / MatUp rows; maps from `deeptex_arms_report.process`, renders from `svbrdf_render_eval` |

## How it works

Inputs:

* **Arm outputs.** `--arm TAG=DIR`, repeated, is any `matsynth/eval_arms.py
  --out` (`DIR/<material>/{five maps}.png` + `sr_manifest.json`):
  `runs/deeptex/eval/matsynth_all` (the reference) and
  `runs/deeptex/eval/render_r{0a,0b,1}_{30,35,40}k`. A `_NNk` tag must match
  the manifest's step; `--expect-step` checks every arm.
* **MatUp baseline.** GT, LR, bilinear and MatUp columns come from
  `runs/matsynth/<material>/` (`--root`) and its newest
  `matup-output-<m>/<stamp>/` (`--run` picks another; `variants/compare/matup_runs.py`),
  through `variants/matup/baseline/matup_render.py` (`load_set`, `crop`,
  `DISP_VAL`, `EVAL_LIGHT`) and `variants/compare/matup_postprocess.py`
  (`deeptex_column`, border, `MAPS`). The renderer is `external/matup_egsr`,
  always at `variants/util/paths.MATUP_ROOT` (`/workspace/external/matup_egsr`
  in the containers; there is no environment override).
* **Lights** from `variants/util/matup_lights.py`: train100 (MatUp's training
  lights), held-out (EVAL_LIGHT + 16 turned 43.25 deg, >= 2.5 deg from every
  training direction), elevation bands.
* **GT validity mask**: training's rule, `|2v - 1| >= 0.1` and unit
  `n_z >= 1e-3` on the stored GT normal, mirrored from `render_loss/render_tf.py`
  (not imported: TF vs torch).

Outputs (each removes its earlier outputs of these names from `--out` first):

| script | `--out` contents |
|---|---|
| `svbrdf_render_eval.py` | `report.json`, `metrics_long.csv`, `summary.md`, `per_material.md`, `render_<mat>.png` (rows = methods; EVAL_LIGHT, grazing, mid, high held-out light, `|pred - GT|` heatmap at a fixed gain), `overview_render_pred.png` |
| `svbrdf_perceptual_eval.py` | `perceptual_report.json`, `perceptual_long.csv`, `perceptual_summary.md` |
| `deeptex_dt_montage.py` | `dt_<material>.png`, `dt_overview_<key>.png` (key = five maps + render), `README.md` with the command that regenerates the folder |

`svbrdf_perceptual_eval.py` imports `svbrdf_render_eval`;
`deeptex_dt_montage.py` imports both that and `variants/compare/*`.
`render_loss/run.ps1` calls the first two: `-Stage eval-render` writes
`runs/deeptex/render_eval/main/` and copies `summary.md` (as
`render_summary.md`), the `render_*.png` sheets and `overview_render_pred.png`
to `runs/montages/deeptex_render_loss/`; `-Stage gate-render -Gate premise`
writes `runs/deeptex/render_eval/premise/`; `-Stage eval-perceptual` writes
`runs/deeptex/render_eval/perceptual/` and copies the summary and report.
`deeptex_dt_montage.py` is run by hand; its output is
`runs/montages/deeptex_render_loss/dt_only/`.

## Run

Through `variants\deeptex\render_loss\run.ps1 -Stage eval-render` /
`-Stage eval-perceptual` (see `../README.md`). Directly (`--out` absolute, never
under `external/`, else exit 2; the matup container starts in
`/workspace/external/matup_egsr`):

    docker compose run --rm matup python /workspace/variants/deeptex/render_loss/eval/svbrdf_render_eval.py \
        --arm matsynth_all=/workspace/runs/deeptex/eval/matsynth_all \
        --arm render_r1_40k=/workspace/runs/deeptex/eval/render_r1_40k \
        --out /workspace/runs/deeptex/render_eval/main

    cd docker/exp
    docker compose run --rm exp python /workspace/variants/deeptex/render_loss/eval/svbrdf_perceptual_eval.py \
        --arm matsynth_all=/workspace/runs/deeptex/eval/matsynth_all \
        --arm render_r0a_40k=/workspace/runs/deeptex/eval/render_r0a_40k \
        --arm render_r1_40k=/workspace/runs/deeptex/eval/render_r1_40k \
        --out /workspace/runs/deeptex/render_eval/perceptual

    docker compose -f docker/compose.yaml run --rm matup python \
        /workspace/variants/deeptex/render_loss/eval/deeptex_dt_montage.py \
        --arm matsynth_all=/workspace/runs/deeptex/eval/matsynth_all \
        --arm render_r1_40k=/workspace/runs/deeptex/eval/render_r1_40k \
        --out /workspace/runs/montages/deeptex_render_loss/dt_only

Common flags: `--material`, `--run`, `--chunk` (lights per render call, default
10). `svbrdf_render_eval.py --no-sheets` skips the PNGs.
`svbrdf_perceptual_eval.py`: `--lpips-lights heldout|sheet`, `--flip-lights
sheet|heldout`, `--device`, `--threads` (`run.ps1 -EvalDevice cpu
-EvalThreads 4` runs it beside a training job). Exit codes: 2 bad argument or a
missing / unreadable map, 3 self-check failed, 4 non-finite render.

## Known behaviour and defects

* **Rendering is unguarded**, as in MatUp: no `n_z >= 1e-3` guard on the
  predicted normals (a PNG-decoded normal can never have `n_z == 0`).
  Offset-removed normals that hit exactly 0 get the smallest positive float32
  and are counted (`nz_zero_fixed`, expected 0).
* **Stale line citations.** `svbrdf_render_eval.py:169` cites `render_tf.py:125-127`
  for `NZ_MIN` / `MASK_MIN_NORM` / `NORMAL_NORM_FLOOR`; they are at
  `render_loss/render_tf.py:123-125`, and the mask is applied at `:368`.
* Docstrings name deleted `scripts/...` paths and `run_deeptex.ps1`
  (`deeptex_dt_montage.py`'s usage line shows `/workspace/scripts/compare/...`;
  the committed `runs/montages/deeptex_render_loss/dt_only/README.md` has that
  path in its regeneration command too). The script itself writes its own path
  (`SCRIPT`) into the next `dt_only/README.md`.
