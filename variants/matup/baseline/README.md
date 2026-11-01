# variants/matup/baseline/

The MatUp baseline: MatUp (`external/matup_egsr`, pinned `c74ec99`) run over the
extracted MatSynth materials, plus the two post-processes the reference
implementation does not have. `run.ps1` is `scripts/matup/run_matup.ps1` moved
here; `variants/tests/l5_matup.ps1` proved the two issue the same docker calls
(deleted with the old code; both are at tag `variants-verified`).

## MatUp itself: unmodified

`main.py` is the submodule's own, called as is inside the `matup` container
(working dir `/workspace/external/matup_egsr`, docker/compose.yaml). Nothing in
`external/matup_egsr` is edited. The call `run.ps1` makes per material:

```
docker compose -f docker/compose.yaml run --rm matup python main.py
    --upscaler swinir --ogl_normals --tile_width 17
    --input_mat       /workspace/runs/matsynth/<m>/SVBRDF/
    --render_dir      /workspace/runs/matsynth/<m>/radiance_LR/
    --upsamplings_dir /workspace/runs/matsynth/<m>/radiance_Upscaled/
    --output_folder   /workspace/runs/matsynth/<m>/matup-output-<m>/
```

* `--ogl_normals` is the one flag that is ours to choose and must not be dropped:
  MatSynth normals are OpenGL, and without it MatUp optimises against a
  Y-inverted surface with no error.
* The three intermediate/output folders are per material because MatUp's
  defaults are shared and `main_optim.py` rmtree's them on success, so two runs
  would delete each other's input.
* A run counts as `ok` only if `radiance_LR/` and `radiance_Upscaled/` were
  removed (MatUp does that only on success) and `...done` was printed; the exit
  code is not used. `nvml error` / `CUDA error: unknown error` / `failed to
  create shim task` abort the batch as GPU loss, `please download model at:`
  as a checkpoint problem, and two consecutive failures abort too.

## Our additions (post-processing, outside the submodule)

| script | container | writes | why |
|---|---|---|---|
| `matup_height.py` | dataprep | `matup-output-<m>/<stamp>/height_opt.png` | paper 3.3 height reconstruction (DCT Poisson integration of the normal map, high-passed, composited with the bilinear LR height); MatUp writes no height |
| `matup_render.py` | matup | `<stamp>/render_lr.png`, `render_gt.png`, `render_bilinear.png` | renders LR, GT and bilinear maps with MatUp's own renderer and eval light `[0.276172, 0.276172, 0.920575]`; MatUp renders only its own output |

The `variants/` copies of these two are moved here by a parallel part of this
refactor; `run.ps1` only orchestrates them. The comparison set and montage scripts they feed
live in `variants/compare/` (`matup_postprocess.py`, `matup_montage.py`).

## Steps and commands

`run_matup.ps1` has no `-Stage` parameter (only `run_deeptex.ps1` does). Its
steps are chosen by switches, in this order:

| step | when | command (`docker compose -f docker/compose.yaml run --rm ...`) |
|---|---|---|
| preflight | unless `-SkipPreflight` | `docker context show` (must be `desktop-linux`), `docker ps --format {{.Names}}` (no containers except `-AllowContainers`), SwinIR checkpoint is 142473939 bytes, `matup python -c "import torch; print(torch.cuda.is_available())"` = `True` |
| per material | each `runs/matsynth/<m>/` with an `SVBRDF/` (or `-Material`); skipped if `matup-output-<m>/**/baseColor_opt.png` exists, unless `-Force` | `matup python main.py ...` (above); log in `runs/_logs/<stamp>/<m>.log` |
| height | `-PostProcess` and no failure | `dataprep python variants/matup/baseline/matup_height.py --all` |
| renders | same | `matup python /workspace/variants/matup/baseline/matup_render.py --all` |
| comparison sets + metrics | same | `dataprep python variants/compare/matup_postprocess.py --all --metrics` |
| montages | same | `dataprep python variants/compare/matup_montage.py --all --collect /workspace/runs/montages` |

The order of the last four is load-bearing: height and render produce the two
columns MatUp does not write, postprocess normalises everything into
`runs/matsynth/<m>/compare/<stamp>/`, montage reads only from there.

Parameters are unchanged: `-Material`, `-Root` (default `runs/matsynth`),
`-Force`, `-SkipPreflight`, `-PostProcess`, `-TileWidth` (default 17),
`-AllowContainers`.

## Path dependencies on other arms

These are why MatUp output paths and file names must not move:

* **Every DeepTextures evaluation reads MatUp's `render_lr.png`.** The DT
  "render" job super-resolves MatUp's LR render
  (`scripts/deeptex/deeptex_compare.py:101`, `("render", "run", "render_lr.png")`),
  taken from the newest `runs/matsynth/<m>/matup-output-<m>/<stamp>/`. That file
  exists only after `matup_render.py` has run, i.e. after `run.ps1 -PostProcess`.
* `render_gt.png` / `render_bilinear.png` / `render_opt.png` in the same folder are
  the references for `scripts/deeptex/render_gate.py` (`[export] ... vs committed
  render_gt.png`) and `scripts/compare/svbrdf_render_eval.py` /
  `svbrdf_perceptual_eval.py`.
* `run_deeptex.ps1` (eval-postprocess, eval-montage) calls the same
  `matup_postprocess.py --deeptex-root ...` and `matup_montage.py`, writing into
  the same `runs/matsynth/<m>/compare/<stamp>/` and `runs/montages/`.
* Every consumer picks the run folder with `sorted(...)[-1]`. A new MatUp run
  (for example `-Force`) therefore changes what all DT evaluations read.
* The DT side's eval root is fixed at `/workspace/runs/matsynth`
  (`run_deeptex.ps1:153`, and every script's `--root` default). `-Root` here only
  redirects `main.py`.

## Changes against `scripts/matup/run_matup.ps1`

| | old | new |
|---|---|---|
| repo root | `Split-Path -Parent (Split-Path -Parent $PSScriptRoot)` | three `Split-Path -Parent` (one level deeper) |
| preflight | `. (Join-Path (Split-Path -Parent $PSScriptRoot) "common\preflight.ps1")` | `. (Join-Path $repo "variants\util\preflight.ps1")` (byte copy of `scripts/common/preflight.ps1`) |
| height | `scripts/matup/matup_height.py` | `variants/matup/baseline/matup_height.py` |
| renders | `/workspace/scripts/matup/matup_render.py` | `/workspace/variants/matup/baseline/matup_render.py` |
| postprocess | `scripts/compare/matup_postprocess.py` | `variants/compare/matup_postprocess.py` |
| montage | `scripts/compare/matup_montage.py` | `variants/compare/matup_montage.py` |
| comments | `.EXAMPLE` paths, a pointer to `common/preflight.ps1` | updated, plus a note saying where the file came from |

Nothing else differs: parameters, console output, log paths, success tests and
abort rules are the same. `run_matup.ps1` passes no DeepTextures-specific
argument anywhere.

## Odd behaviour, kept (variants/README.md rule 5)

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
* `preflight.ps1` (copied byte for byte) still defines the unused
  `Test-NoStrayContainers`, and its comment points at `run_matup.ps1:108-123`,
  which is now lines 106-119 of the old file.

## Verification

Done and removed with the old code. `variants/tests/l5_matup.ps1` (tag `variants-verified`)
ran both orchestrators from byte copies in sandbox repo roots with `docker` replaced by a
recording function, over 13 scenarios (single, batch, skip-done, force, postprocess,
postprocess-blocked, two-failures, gpu-lost, checkpoint, root-tilewidth, unknown-material,
preflight, preflight-stray), and compared docker argv after the path mapping above, the
normalised transcript and the created files: all PASS (last run 20261101, report
`runs/variants_verify/matup/20261101-024030/report.md`).
