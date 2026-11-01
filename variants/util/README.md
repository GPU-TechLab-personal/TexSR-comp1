# variants/util/

Helpers that more than one method or experiment needs and that belong to none of
them: where the submodules are, how a MatSynth material is extracted and
encoded, MatUp's light tables, the shared metrics, and the PowerShell helpers
every `run.ps1` dot-sources. Nothing here runs an experiment by itself.

## Contents

| file | what it does |
|---|---|
| `paths.py` | Repo and submodule locations (`REPO`, `DEEPTEX_ROOT`, `MATUP_ROOT`, `DONOR_ROOT`, ...) and the only place a submodule goes onto `sys.path`: `use_deeptex()`, `use_matup()`, `use_swinir()`; `check_pin()` checks the submodule commit. `DEEPTEX_ROOT` can be overridden by the environment variable of the same name. |
| `matsynth_extract.py` | Extracts one MatSynth material from a HuggingFace parquet shard into the layout MatUp reads: `<name>/SVBRDF/*.png` (128², the LR input), `<name>/GT/*.png` (512²), `meta.json`. Run in the dataprep container. |
| `map_encode.py` | `to_rgb8()`: any MatSynth map -> the 8-bit RGB a DeepTextures network takes. Used by both the training pairs (`deeptex/matsynth/matsynth_make_pairs.py`) and the evaluation inputs (`deeptex/infer/deeptex_compare.py`, `matsynth/eval_arms.py`), so the two cannot drift apart. numpy + cv2 only. |
| `matup_lights.py` | MatUp's Fibonacci training lights and the held-out evaluation lights, in numpy (Python 3.8 clean). Used by the render loss (`deeptex/render_loss/render_tf.py`, TF image without torch), its parity gate and the render eval (matup image). |
| `texsr_metrics.py` | Metrics shared by both methods: `sharpness()` / `corr()` (the reference-free numbers behind the committed `runs/montages/`), `psnr()` / `ssim()` / `full_reference()` on the 8-bit ladder. numpy + cv2 only, so every container computes the same number. |
| `arms.py` | The arm registry `variants/arms.json`: `display_name`, `column_key`, `method_display`, `reference`, `pairs` / `first_pair` (head-to-head), `split_tag` (`<base>_<NN>k`). stdlib only. |
| `arm_output.py` | Writing an SR arm's evaluation folder, for any method: the folder/manifest contract (docstring), `load_as_rgb01`, `save_sr` (+ `COLLAPSE`), `ALL_JOBS`, `tile_grid` / `super_resolve(predict, ...)`, `write_arm`. numpy + cv2 only. |
| `report/` | Comparison reports for any experiment: `images` (I/O, 16-bit promotion, display, `fit_to`/`load_fit`), `metrics` (sharpness, corr, PSNR/SSIM vs GT), `sheets` (strip / grid / table, `fit`, `placeholder`), `layout` (`<m>/compare/<run>/<map>/<column>.png`), `montage` (grids and overviews read from a comparison set), `overview.py` (CLI), `drift` (colour drift per checkpoint, model-agnostic). See `report/__init__.py`. |
| `lib.ps1` | Functions every DeepTextures `run.ps1` shares: `Invoke-Step` (one docker step, logged; `-NamePrefix` / `$script:StepNamePrefix` for another method's containers), `Get-FailureKind`, `Get-ArmInfo` / `Get-DisplayName` (from `arms.json`), checkpoint lookup and `Start/Stop-CheckpointKeeper`, `Get-CompleteEval` (reuses an eval folder whose manifests match the step and whose checkpoint file - `checkpoint_file`, else `<checkpoint>.index` - still exists), `Invoke-CompareAtSteps`, `Read-CfgValue`, the render-gate readers. Reads the run.ps1's path variables from script scope. |
| `preflight.ps1` | Host checks shared by the DeepTextures and MatUp orchestrators. `Invoke-SharedPreflight`: docker context must be `desktop-linux` (`Test-DockerContext`), the daemon must answer, and no other container may be running except those passed as `-AllowContainers`. |

## How it works

- Every Python entry point under `variants/` starts with the same bootstrap and
  then asks `paths` for the submodule it needs:

  ```python
  import os, sys
  sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * N)))
  from variants.util import paths      # N = depth of the file below the repo root
  paths.use_deeptex()                  # or use_matup() / use_swinir()
  ```

  Inside the containers the repo is `/workspace` (`docker/compose.yaml`,
  `docker/exp/compose.yaml`).
- `lib.ps1` is dot-sourced after a `run.ps1`'s parameters and path variables
  (`. (Join-Path $repo "variants\util\lib.ps1")`); it dot-sources `preflight.ps1`
  itself. `matup/baseline/run.ps1` dot-sources `preflight.ps1` directly.

## Run

```
docker compose run --rm dataprep python variants/util/matsynth_extract.py --list
docker compose run --rm dataprep python variants/util/matsynth_extract.py --index 0 --count 10
```

(from `docker/`; the other files are libraries.)

## Known behaviour

- `paths.py`'s docstring names a `use_donor()` that does not exist; the donor
  (`external/esrgan-tf2`) is only reachable as `DONOR_ROOT` and is never put on
  `sys.path` (its `modules` package would shadow DeepTextures').
- `preflight.ps1` defines `Test-NoStrayContainers`, which nothing calls.
- `sharpness()` / `corr()` must stay bit-identical: the committed MatUp numbers
  under `runs/montages/` were produced by them (docs/RUNBOOK.md).
