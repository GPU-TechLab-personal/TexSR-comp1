# compare — what both methods share

The LR / bilinear / MatUp / GT columns every comparison is built on, and nothing
method-specific: the DeepTextures column and sheet of the paper arm are
`deeptex/paper/deeptex_{postprocess,montage}.py`, which call these with a hook.

| file | old file | kind |
|---|---|---|
| `matup_postprocess.py` | `scripts/compare/matup_postprocess.py` | `process`, `main` decomposed; every other def AST-identical |
| `matup_montage.py` | `scripts/compare/matup_montage.py` | `process`, `collect`, `main` decomposed; `build` and the rest AST-identical |
| `matup_overview.py` | `scripts/compare/matup_overview.py` | moved, byte-identical code |
| `matup_runs.py` | the four `newest_run` copies | merged (see below) |

## What changed against the old code (new file:line)

| new | was | change |
|---|---|---|
| `matup_postprocess.py:53-55` | `matup_postprocess.py:46-48` | bootstrap; `texsr_metrics` from `variants.util`, `newest_run` from `matup_runs` |
| `matup_postprocess.py:78-80` | `matup_postprocess.py:71-82` | `DEEPTEX_COLUMN` / `DEEPTEX_MAPS` / `ORDER_WITH_DEEPTEX` moved to `deeptex/paper/deeptex_postprocess.py` |
| `matup_postprocess.py:257-399` (`process(mat_dir, args, extra=None)`) | `matup_postprocess.py:258-414` | `newest_run` instead of the inline glob (:260-267); the column order and the five DeepTextures spots are `extra.*` calls (:268, :307, :335, :356, :387, :394) |
| `matup_postprocess.py:402-431` | `matup_postprocess.py:417-443` | `parser()`, `run(args, extra)`, `main()`; `--deeptex-root` moved to the paper arm |
| `matup_montage.py:51` | `matup_montage.py:53-61` | `DEEPTEX_*` moved to `deeptex/paper/deeptex_montage.py` |
| `matup_montage.py:140-167` (`process(..., extra=None)`) | `matup_montage.py:150-186` | the `montage_deeptex.png` branch (:173-185) is `extra.after_montage` |
| `matup_montage.py:173-239` (`collect(..., extra=None)`) | `matup_montage.py:192-268` | the `_deeptex.png` copy and note are `extra.collect_one` / `count_note` / `note_lines` |
| `matup_montage.py:242-273` | `matup_montage.py:271-296` | `parser()`, `run(args, extra)`, `main()` |

Without a hook these write exactly what the old scripts wrote without a DeepTextures
column (L3 `base`: old vs new byte-identical); with the paper hooks, exactly what the
old scripts wrote with it (L3 `montages`, against the committed `runs/montages/`).

## newest_run, once

`matup_runs.newest_run(mat_dir, name, want)` is `sorted(isdir entries of
<mat>/matup-output-<mat>/*)[-1]`, filtered to one stamp when asked. It replaces
`deeptex_compare.py:183-194`, `svbrdf_render_eval.py:450-456`,
`deeptex_arms_report.py:104-109` and the inline copy in `matup_postprocess.py:260-267`
(L0 `decomposed` entries). Not merged, because they are not this area's to change:
`render_gate.py:378-386` (gates) and the inline copies in `matup/baseline/matup_render.py:165-172`
and `matup_height.py:113-115` (moved verbatim).

## Path dependencies on other arms

- Everything here reads the MatUp baseline's tree: `runs/matsynth/<m>/{SVBRDF,GT}`,
  `matup-output-<m>/<stamp>/{*_opt.png, height_opt.png, render_*.png}` (`height_opt.png`
  from `matup/baseline/matup_height.py`, `render_*.png` from `matup_render.py`), and writes
  `runs/matsynth/<m>/compare/<stamp>/`; `--collect` writes the committed `runs/montages/`.
- Imported by `deeptex/matsynth/deeptex_arms_report.py`, `deeptex/render_loss/eval/*`,
  `deeptex/matsynth/deeptex_drift.py` (`to_display`, `placeholder`, `fit`, `deeptex_column`, ...).

## Kept, not fixed

- `matup_montage.collect()` takes each material's newest `compare/*/montage.png` (:189)
  even when `process()` was told `--run <stamp>`, so `--run` with `--collect` can collect a
  different stamp than it drew.
- `matup_overview.py` has a DeepTextures column built in (`COLUMNS`); without a deeptex.png
  it draws `n/a`. It is the two-method overview, so it stays here unchanged.
- The `summary.md` text still says `scripts/matup/matup_height.py` (it is committed output).
- A run of the plain `compare/matup_montage.py` on a `compare/<stamp>/` that already has
  deeptex.png columns no longer draws `montage_deeptex.png` (the old script did, by file
  existence); that sheet is the paper arm's (`deeptex/paper/deeptex_montage.py`).
