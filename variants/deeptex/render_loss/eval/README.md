# deeptex/render_loss/eval — the predicted SVBRDFs, judged in render space

| file | old file | kind |
|---|---|---|
| `svbrdf_render_eval.py` | `scripts/compare/svbrdf_render_eval.py` | every def AST-identical except `newest_run` (merged into `compare/matup_runs.py`) |
| `svbrdf_perceptual_eval.py` | `scripts/compare/svbrdf_perceptual_eval.py` | moved (L0: AST-identical) |
| `deeptex_dt_montage.py` | `scripts/compare/deeptex_dt_montage.py` | moved (L0: AST-identical; `MATUP_ROOT`, `SCRIPT` allowed) |

## What changed against the old code (new file:line)

| new | was | change |
|---|---|---|
| `svbrdf_render_eval.py:152-155` | `svbrdf_render_eval.py:151-157` | bootstrap; `REPO = paths.REPO`; `MATUP_ROOT = paths.use_matup()` replaces the `MATUP_ROOT` env default and the four `sys.path` inserts |
| `svbrdf_render_eval.py:160-165` | `svbrdf_render_eval.py:162-166` | `matup_lights` (common), `matup_montage`/`overview`/`postprocess` (compare), `matup_render` (matup/baseline), `newest_run` (compare/matup_runs) |
| `svbrdf_render_eval.py:449-450` | `svbrdf_render_eval.py:450-456` | `newest_run` removed (same body in `compare/matup_runs.py`; `sre.newest_run` / `ev.newest_run` still resolve) |
| `svbrdf_perceptual_eval.py:126-131` | `svbrdf_perceptual_eval.py:125-131` | bootstrap; `svbrdf_render_eval` by package |
| `deeptex_dt_montage.py:68-78` | `deeptex_dt_montage.py:67-80` | bootstrap; `MATUP_ROOT = paths.use_matup()`; siblings by package |
| `deeptex_dt_montage.py:80` | `deeptex_dt_montage.py:82` | `SCRIPT` = the new path - it is printed into `dt_only/README.md`'s regeneration command, so that line differs from the committed README by design |

**Behaviour change:** the old `MATUP_ROOT` environment override (svbrdf_render_eval.py:153,
deeptex_dt_montage.py:68) is gone; MatUp is always `paths.MATUP_ROOT`
(`/workspace/external/matup_egsr` in the containers, as before by default).

L3 runs old and new `svbrdf_render_eval` and `svbrdf_perceptual_eval` on the committed
arm folders, on the GPU as committed, and compares with `runs/montages/deeptex_render_loss/
{render_summary.md,perceptual_report.json,perceptual_summary.md}` and
`runs/deeptex/render_eval/main/*` (`l3_reports.py render,perceptual`).

## Path dependencies on other arms

- **Arm outputs.** `--arm TAG=DIR` is any `matsynth/eval_arms.py --out`:
  `runs/deeptex/eval/matsynth_all` (the reference) and `render_r{0a,0b,1}_{30,35,40}k`.
  A `_NNk` tag must match the manifest's step.
- **MatUp baseline.** GT, LR, bilinear and MatUp columns come from `runs/matsynth/<m>/`
  and its newest `matup-output-<m>/<stamp>/` through `matup/baseline/matup_render.py`
  (`load_set`, `crop`, `DISP_VAL`, `EVAL_LIGHT`) and `compare/matup_postprocess.py`
  (`deeptex_column`, border, `MAPS`); the renderer is `external/matup_egsr`.
- **Training.** The GT validity mask is training's rule, mirrored from `render_loss/render_tf.py`
  (not imported: TF vs torch).
- `svbrdf_perceptual_eval.py` imports `svbrdf_render_eval` (its renders, lights, columns);
  `deeptex_dt_montage.py` imports both plus `compare/*`.

## Kept, not fixed

- **Stale line citations.** `svbrdf_render_eval.py:169` cites `render_tf.py:125-127` for
  `NZ_MIN` / `MASK_MIN_NORM` / `NORMAL_NORM_FLOOR`: they are at `scripts/deeptex/render_tf.py:128-130`
  (`variants/deeptex/render_loss/render_tf.py:123-125`). `:52`, `:169` and `:442` cite
  `render_tf.py:366-371` for the mask; it is applied at `scripts/deeptex/render_tf.py:373`
  (`:366-371` is the baseColor/normal decode).
- Docstrings of moved files cite line numbers of the OLD files (`scripts/...`).
