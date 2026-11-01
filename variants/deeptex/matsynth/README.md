# deeptex/matsynth — the MatSynth arms (all / chratio / basecolor)

The released trainer, unmodified, on MatSynth pairs instead of textures_db, scored on
every map and the render. Training is `upstream_runner/run_upstream_train.py` with one
of the three configs here.

| file | old file | kind |
|---|---|---|
| `matsynth_make_pairs.py` | `scripts/deeptex/matsynth_make_pairs.py` | moved (L0: AST-identical) |
| `recipes.py` | `texsr_build_tfrecord.py --repeat` + `run_deeptex.ps1:539-543` | split out (decomposed) |
| `eval_arms.py` | `deeptex_compare.py --maps all --scalar-collapse` | split out (decomposed) |
| `deeptex_arms_report.py` | `scripts/deeptex/deeptex_arms_report.py` | every def AST-identical except `newest_run` (merged) |
| `deeptex_drift.py` | `scripts/deeptex/deeptex_drift.py` | moved (L0: AST-identical) |
| `esrgan_matsynth{,_chratio,_basecolor}.yaml` | `configs/deeptex/` | byte-identical copies |

## The three recipes

`recipes.py:39-43`, the old `$script:MatsynthRecipes`: how many times each map folder
is packed into the ONE record (one seeded shuffle over all of it, because upstream
reads with `shuffle=False` and the record order is the curriculum).

| recipe | baseColor | normal | roughness | metallic | height | config |
|---|---|---|---|---|---|---|
| `all` (the paper's mix) | 1 | 1 | 1 | 1 | 1 | `esrgan_matsynth.yaml` |
| `chratio` (ablation, channel count) | 3 | 3 | 1 | 1 | 1 | `esrgan_matsynth_chratio.yaml` |
| `basecolor` (control) | 1 | - | - | - | - | `esrgan_matsynth_basecolor.yaml` |

The three configs differ from `paper/esrgan_paper.yaml` only in their header comment and
`train_dataset.path` (`.../datasets/matsynth/matsynth_<recipe>_lrx4.tfrecord`).

## What changed against the old code (new file:line)

| new | was | change |
|---|---|---|
| `matsynth_make_pairs.py:75-79` | `matsynth_make_pairs.py:75-78` | bootstrap; `map_encode`, `matsynth_extract` from `variants.util` |
| `recipes.py:39-43` | `run_deeptex.ps1:539-543` | the recipe table moves into Python |
| `recipes.py:55-68` | `texsr_build_tfrecord.py:86-88, 102-105` | `--repeat` and its check; calls `upstream_runner/texsr_build_tfrecord.build(args, repeats)` |
| `eval_arms.py:50-57` | `deeptex_compare.py:95-102` | all six jobs (the core has baseColor and render only) |
| `eval_arms.py:59-66` | `deeptex_compare.py:148-155` | `COLLAPSE` |
| `eval_arms.py:69-88` | `deeptex_compare.py:158-180` | `save_sr`: the scalar branch here, the colour branch is the core's `save_colour` |
| `eval_arms.py:91-100` | `deeptex_compare.py:206-213, 274, 286-287` | `--maps` default `all`, `--scalar-collapse`, the collapse recorded in the manifest |
| `eval_arms.py:45-46` | — | re-exports `load_as_rgb01`, `newest_run` for `deeptex_drift.py` |
| `deeptex_arms_report.py:68-73` | `deeptex_arms_report.py:67-71` | bootstrap; `compare/*` by package; `newest_run` from `compare/matup_runs.py` (was `:104-109`) |
| `deeptex_drift.py:48-64` | `deeptex_drift.py:47-62` | bootstrap; `eval_arms` imported **as `deeptex_compare`** (it provides the four names the old file read from there), `deeptex_sr` from `infer/` |

## Path dependencies on other arms

- **Eval set.** `matsynth_make_pairs.py --check-eval` rebuilds the 12 evaluation materials
  from `/data/MatSynth/data/test-00000-of-00008.parquet` and requires pixel identity
  with `runs/matsynth` (built by `common/matsynth_extract.py`); the pairs exclude those
  materials by name.
- **Pairs.** `runs/deeptex/datasets/matsynth/<map>/{HR,LRx4}` feed `recipes.py` here AND
  the grouped record of `render_loss/build/` (same folders, same `_meta/`).
- **MatUp baseline.** `eval_arms.py`'s `render` job super-resolves MatUp's
  `render_lr.png` (newest `matup-output-<m>/<stamp>/`, from `matup/baseline/matup_render.py`).
  `deeptex_arms_report.py` takes its lr / bilinear / MatUp / GT columns from
  `runs/matsynth/<m>/` through `compare/matup_postprocess.py` (same crop, border and
  DirectX->OpenGL flip as `runs/montages/`).
- **Other arms' outputs.** `deeptex_arms_report.py --arm` reads any `eval_arms.py --out`:
  the textures_db baseline is `runs/deeptex/eval/paper_allmaps` (the paper checkpoint run
  through `eval_arms.py`), the render-loss arms are `runs/deeptex/eval/render_r*_<k>k`.
- `deeptex_drift.py` reads `runs/deeptex/datasets/matsynth/<map>` (train pairs) and
  `runs/matsynth/<m>/{SVBRDF,GT}`.

## Kept, not fixed

- `esrgan_matsynth_chratio.yaml`'s header still says the record is "built with
  texsr_build_tfrecord.py --repeat" and gives a `configs/deeptex/...` path; the
  files are byte-identical copies (the ps1 reads `sub_name` with a line regex).
- `deeptex_drift.py` hard-codes scale 4, whole-image inference (`super_resolve(..., 4, 0, 16)`)
  and the `"mean"` collapse for its sheet, whatever the config says.
- Docstrings of moved files cite line numbers of the OLD files (`scripts/...`).

## run.ps1

Stages `prep-matsynth` (`-Recipe all|chratio|basecolor`) and `eval-arms`, plus
`preflight`; `-Stage` is mandatory (the old default `all` is the paper arm's).
Generated from `scripts/deeptex/run_deeptex.ps1`: each stage function below is the old one verbatim except the script/config paths (old line ranges in the file header). Shared helpers come from `variants/util/lib.ps1`. L5 (`variants/tests/l5_deeptex.ps1`) runs every stage through both and compares the docker argv, the console transcript and the files created.
Changed: the record build is one `recipes.py --recipe <R> --root <ms>` call
instead of the per-map `--hr/--lr[/--repeat k]` list, and the `recipe <R> : ...`
console line is printed by `recipes.py` (L5 rule R checks that the old list is
exactly the recipe table). `eval-arms` calls `eval_arms.py` without `--maps all
--scalar-collapse mean`, its defaults.

Kept as it was: `-PaperConfig` / `-PaperRoot` still default to the paper arm's
(`variants/deeptex/paper/esrgan_paper.yaml`, `runs/deeptex/paper`), so
`-Stage eval-arms` without them evaluates the textures_db checkpoint under the tag
`paper`.
