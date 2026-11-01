# variants/matup/baseline/

The MatUp baseline arm: MatUp (`external/matup_egsr`, pinned `c74ec99`, not
edited) run over the extracted MatSynth materials in `runs/matsynth/`, plus the
two post-processes the reference implementation does not have (height map,
LR/GT/bilinear renders). Its output is also an input of every DeepTextures
evaluation.

## Contents

| file | container | what it does |
|---|---|---|
| `run.ps1` | host | orchestrator: preflight, MatUp `main.py` per material (resumable), then the optional post-processing chain |
| `matup_height.py` | dataprep | paper 3.3 height reconstruction: DCT (Neumann) Poisson integration of MatUp's normal map over the valid region, high-passed and composited with the bilinear LR height; writes `matup-output-<m>/<stamp>/height_opt.png` (MatUp writes no height) |
| `matup_render.py` | matup | renders the LR, GT and bilinear maps with MatUp's own renderer and eval light `[0.276172, 0.276172, 0.920575]`; writes `<stamp>/render_lr.png`, `render_gt.png`, `render_bilinear.png` (MatUp renders only its own output). `--verify` re-renders MatUp's maps against its `render_opt.png` |

Shared code it calls: `variants/util/preflight.ps1` (docker context and stray
container checks), `variants/compare/matup_postprocess.py` and
`variants/compare/matup_montage.py` (comparison sets, metrics, montages).

## How it works

Input: every `runs/matsynth/<m>/` with an `SVBRDF/` folder (the LR maps), or one
`-Material`. Per material, `run.ps1` calls the submodule's own `main.py`
unchanged inside the `matup` container (working dir
`/workspace/external/matup_egsr`, `docker/compose.yaml`):

```
docker compose -f docker/compose.yaml run --rm matup python main.py
    --upscaler swinir --ogl_normals --tile_width 17
    --input_mat       /workspace/runs/matsynth/<m>/SVBRDF/
    --render_dir      /workspace/runs/matsynth/<m>/radiance_LR/
    --upsamplings_dir /workspace/runs/matsynth/<m>/radiance_Upscaled/
    --output_folder   /workspace/runs/matsynth/<m>/matup-output-<m>/
```

* `--ogl_normals` must not be dropped: MatSynth normals are OpenGL, and without
  it MatUp optimises against a Y-inverted surface with no error.
* The three intermediate/output folders are per material because MatUp's
  defaults are shared and `main_optim.py` rmtree's them on success, so two runs
  would delete each other's input.
* A run counts as `ok` only if `radiance_LR/` and `radiance_Upscaled/` were
  removed (MatUp does that only on success) and `...done` was printed; the exit
  code is not used. `nvml error` / `CUDA error: unknown error` / `failed to
  create shim task` abort the batch as GPU loss, `please download model at:`
  as a checkpoint problem, and two consecutive failures abort too.
* The console log of each run goes to `runs/_logs/<stamp>/<m>.log`; a summary
  table (material, status, seconds, final loss) is printed at the end.

Output: `runs/matsynth/<m>/matup-output-<m>/<stamp>/` (MatUp's `*_opt.png`,
`render_opt.png`; plus `height_opt.png` and the three renders after
post-processing), then `runs/matsynth/<m>/compare/<stamp>/` and `runs/montages/`.

### Steps

There is no `-Stage` parameter; switches pick the steps, which run in this order:

| step | when | command (`docker compose -f docker/compose.yaml run --rm ...`) |
|---|---|---|
| preflight | unless `-SkipPreflight` | `docker context show` (must be `desktop-linux`), `docker ps --format {{.Names}}` (no containers except `-AllowContainers`), SwinIR checkpoint is 142473939 bytes, `matup python -c "import torch; print(torch.cuda.is_available())"` = `True` |
| per material | each material; skipped if `matup-output-<m>/**/baseColor_opt.png` exists, unless `-Force` | `matup python main.py ...` (above) |
| height | `-PostProcess` and no failure | `dataprep python variants/matup/baseline/matup_height.py --all` |
| renders | same | `matup python /workspace/variants/matup/baseline/matup_render.py --all` |
| comparison sets + metrics | same | `dataprep python variants/compare/matup_postprocess.py --all --metrics` |
| montages | same | `dataprep python variants/compare/matup_montage.py --all --collect /workspace/runs/montages` |

The order of the last four is load-bearing: height and render produce the two
columns MatUp does not write, postprocess normalises everything into
`runs/matsynth/<m>/compare/<stamp>/`, montage reads only from there.

### Who reads this output

These are why MatUp output paths and file names must not move:

* **Every DeepTextures evaluation reads MatUp's `render_lr.png`.** The DT
  "render" job super-resolves MatUp's LR render (`JOBS` in
  `variants/deeptex/infer/deeptex_compare.py` and
  `variants/deeptex/matsynth/eval_arms.py`, `("render", "run", "render_lr.png", ...)`),
  taken from the newest `runs/matsynth/<m>/matup-output-<m>/<stamp>/`. That file
  exists only after `matup_render.py` has run, i.e. after `run.ps1 -PostProcess`.
* `render_gt.png` / `render_bilinear.png` / `render_opt.png` in the same folder
  are the references for `variants/deeptex/render_loss/gates/render_gate.py` and
  `variants/deeptex/render_loss/eval/svbrdf_render_eval.py` /
  `svbrdf_perceptual_eval.py`.
* `variants/deeptex/paper/run.ps1 -Stage eval` (eval-postprocess, eval-montage)
  runs `deeptex_postprocess.py` (which is `compare/matup_postprocess.py` plus a
  fifth column) and `deeptex_montage.py`, writing into the same
  `runs/matsynth/<m>/compare/<stamp>/` and `runs/montages/`.
* Every consumer picks the run folder with `sorted(...)[-1]`
  (`variants/compare/matup_runs.py:newest_run`, plus inline copies in
  `matup_height.py`, `matup_render.py` and `render_gate.py`). A new MatUp run
  (for example `-Force`) therefore changes what all DT evaluations read.
* The DT side's eval root is fixed at `/workspace/runs/matsynth` (`$EVAL_ROOT`
  in the DT `run.ps1` files, and every script's `--root` default). `-Root` here
  only redirects `main.py`.

## Run

From the repo root on the Windows host (Docker Desktop, `desktop-linux` context):

```
variants\matup\baseline\run.ps1
variants\matup\baseline\run.ps1 -Material acg_metal_026
variants\matup\baseline\run.ps1 -Force -PostProcess
```

Parameters: `-Material`, `-Root` (default `runs/matsynth`), `-Force`,
`-SkipPreflight`, `-PostProcess`, `-TileWidth` (default 17),
`-AllowContainers`. No DeepTextures-specific argument is passed anywhere.

## Known behaviour and defects

* `-Root` and `-TileWidth` reach `main.py` only. The four post-processing steps
  always run with their defaults (`--root /workspace/runs/matsynth`,
  `--tile-width 17`), so `-Root runs/other -PostProcess` post-processes
  `runs/matsynth`, and with a non-default tile width `matup_height.py` and
  `matup_render.py` still crop a 17-wide border (`matup_postprocess.py` only warns
  that the measured border disagrees).
* Post-processing steps are not checked: a failing step prints its error and the
  chain continues. The script never sets a non-zero exit code for failed
  materials either; only a `throw` (validation, preflight) does.
* The post-processing output filter drops every line matching
  `UserWarning|_VF\.meshgrid`, not only the meshgrid warning.
* `-Root` must be relative with forward slashes: it is joined to the repo root
  for host paths and string-concatenated after `/workspace/` for container paths.
* The log is written only after the container exits; an interrupted host script
  leaves no log.
* "Already done" means any `baseColor_opt.png` anywhere under
  `matup-output-<m>/`, whichever run wrote it.
* `variants/util/preflight.ps1` still defines `Test-NoStrayContainers`, which
  nothing calls.
