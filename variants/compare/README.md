# variants/compare/

The shared comparison/report layer over the MatUp baseline's output tree: metrics,
comparison montages with LR / bilinear / MatUp / GT columns, the overview sheet, and
the newest-MatUp-run lookup. The DeepTextures arms plug their column into it through
hooks (`deeptex/paper/deeptex_{postprocess,montage}.py`) or import its column-building
and drawing code for their own reports. It runs no experiment of its own.

## Contents

| file | what it does |
|---|---|
| `matup_postprocess.py` | dataprep container: one MatUp run -> comparison-ready LR / bilinear / MatUp / GT columns per map and render (valid-region crop, normal G flip DirectX -> OpenGL, everything at the valid-region size, 496^2 at border 8, 16-bit), `sheet_<map>.png`, and with `--metrics` `metrics.json`; also the shared helpers `deeptex_column`, `sharpness`, `full_reference`, `read`, `write` |
| `matup_montage.py` | dataprep container: one 4x6 `montage.png` per material (rows LR / bilinear / MatUp / GT, columns five maps + render) from `compare/<stamp>/`; `--collect <dir>` copies every montage and `metrics.json` and writes `summary.md` (MatUp / bilinear gradient ratios); also `build`, `to_display`, `placeholder` |
| `matup_overview.py` | dataprep container: one sheet for a map over all materials (rows) x LR / bilinear / MatUp / DeepTextures / GT (columns), `n/a` where a cell is missing; also `fit`, `to_display` |
| `matup_runs.py` | `newest_run(mat_dir, name, want)`: the newest `<mat>/matup-output-<mat>/<stamp>/` (`sorted(...)[-1]` over its directories), or the one named `want` |

## How it works

```
runs/matsynth/<m>/SVBRDF/*.png (LR 128^2), GT/*.png (512^2)
runs/matsynth/<m>/matup-output-<m>/<stamp>/      newest_run() picks <stamp>
    *_opt.png                (MatUp)
    height_opt.png           (variants/matup/baseline/matup_height.py)
    render_{lr,bilinear,opt,gt}.png  (variants/matup/baseline/matup_render.py)
  -> matup_postprocess.py --all --metrics
       runs/matsynth/<m>/compare/<stamp>/<map>/{lr,bilinear,matup,gt}.png
                                         render/{lr,bilinear,matup,gt}.png
                                         sheet_<map>.png, sheet_render.png, metrics.json
  -> matup_montage.py --all [--collect /workspace/runs/montages]
       compare/<stamp>/montage.png
       runs/montages/<m>.png, <m>.json, summary.md      (committed)
  -> matup_overview.py --map <map> --out runs/montages/overview_<map>.png
```

`process()` detects the black border MatUp leaves (`tile_width // 2`, measured and
checked against `--tile-width`, default 17) and crops every column to the valid
region: `bilinear` is the whole LR map upsampled INTER_LINEAR x4 then cropped (bit for
bit MatUp's initialisation), `lr` is the cropped LR shown with INTER_NEAREST, all
written 16-bit. `metrics.json` per map: `|grad|` and Laplacian variance
per column, `matup_over_bilinear`, `vs_gt` (PSNR / SSIM via `variants/util/texsr_metrics`),
the normal's Y correlation before / after the flip, and the same for the render.

Hooks: `matup_postprocess.run(args, extra)` / `process(mat_dir, args, extra)` call
`extra.order`, `.add_map`, `.add_render`, `.add_metrics`, `.ratio_note`, `.psnr_note`;
`matup_montage.run(args, extra)` calls `extra.after_montage`, `.collect_one`,
`.count_note`, `.note_lines`. Without a hook they write the four MatUp-side columns
only. The paper arm's hooks are `DeeptexColumn` and `DeeptexSheet`.

Who uses it:

- `variants/matup/baseline/run.ps1 -PostProcess` runs `matup_height.py`,
  `matup_render.py`, then `matup_postprocess.py --all --metrics` and
  `matup_montage.py --all --collect /workspace/runs/montages`.
- `variants/deeptex/paper/run.ps1 -Stage eval` runs the same two through the
  DeepTextures hooks.
- Imported by `deeptex/matsynth/deeptex_arms_report.py`, `deeptex/matsynth/deeptex_drift.py`,
  `deeptex/render_loss/eval/{svbrdf_render_eval,deeptex_dt_montage}.py` (columns, crops,
  `to_display`, `placeholder`, `fit`, `deeptex_column`) and `deeptex/infer/deeptex_compare.py`
  (`newest_run`). Following `newest_run` keeps every DeepTextures column on the MatUp
  run the other columns came from.

## Run

From the repo root, in the dataprep container:

```
docker compose run --rm dataprep python variants/compare/matup_postprocess.py --all --metrics
docker compose run --rm dataprep python variants/compare/matup_montage.py --all --collect /workspace/runs/montages
docker compose run --rm dataprep python variants/compare/matup_overview.py \
    --map baseColor --out /workspace/runs/montages/overview_baseColor.png
```

Common options: `--root` (default `/workspace/runs/matsynth`), `--material <m>` or
`--all`, `--run <stamp>` (default the newest). `matup_postprocess.py`: `--tile-width`,
`--no-sheet`, `--metrics`. `matup_montage.py`: `--size` (cell, 320), `--collect`.
`matup_overview.py`: `--map`, `--size` (190), `--out`.

## Known behaviour and defects

- `matup_montage.collect()` takes each material's newest `compare/*/montage.png`
  (`:189`) even when told `--run <stamp>`, so `--run` with `--collect` can collect a
  different stamp than it drew. `matup_overview.py` likewise always reads the newest
  `compare/<stamp>/`.
- `matup_overview.py` has the DeepTextures column built in (`COLUMNS`); without a
  `deeptex.png` it draws `n/a`.
- Plain `matup_montage.py` never draws `montage_deeptex.png`, even on a
  `compare/<stamp>/` that has `deeptex.png` columns; that sheet comes from
  `deeptex/paper/deeptex_montage.py`.
- The `summary.md` text written by `collect()` names `scripts/matup/matup_height.py`
  (now `variants/matup/baseline/matup_height.py`), kept so the committed
  `runs/montages/summary.md` does not change. The `matup_montage.py` docstring cites the
  same `scripts/matup/...` paths.
- `newest_run`'s rule also exists as inline copies in
  `variants/deeptex/render_loss/gates/render_gate.py`,
  `variants/matup/baseline/matup_render.py` and `variants/matup/baseline/matup_height.py`;
  a change to the rule has to be made there too.
- `matup_postprocess.py` refuses a run with a missing `*_opt.png` (except height) and
  skips a material with no MatUp run (`SKIP (no MatUp output)`).
