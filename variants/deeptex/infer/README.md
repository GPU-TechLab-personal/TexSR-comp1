# deeptex/infer — running a checkpoint

| file | old file | kind |
|---|---|---|
| `deeptex_sr.py` | `scripts/deeptex/deeptex_sr.py` | moved (L0: AST-identical) |
| `deeptex_compare.py` | `scripts/deeptex/deeptex_compare.py` | `main`, `save_sr` decomposed; `newest_run` -> `compare/matup_runs.py` |

`deeptex_compare.py` is the shared core: the two maps an RGB network needs no
invented choice for, baseColor and render (the old default `--maps baseColor,render`).
The all-maps mode with its 3->1 collapse is an arm of its own:
`matsynth/eval_arms.py`, which goes through `run()` here with its own job list and
writer.

## What changed against the old code (new file:line)

| new | was | change |
|---|---|---|
| `deeptex_sr.py:63-65` | `deeptex_sr.py:63-64` | bootstrap; `SUBMODULE = paths.use_deeptex()` |
| `deeptex_compare.py:41-52` | `deeptex_compare.py:78-90` | bootstrap; `map_encode`, `deeptex_sr`, `newest_run` by package |
| `deeptex_compare.py:57-60` | `deeptex_compare.py:95-102` | JOBS = baseColor, render (the other four: `matsynth/eval_arms.py:50-57`) |
| `deeptex_compare.py:66` | `deeptex_compare.py:211, 286-287` | `RECORDED_COLLAPSE`: the manifest keeps `"scalar_collapse": "mean"` and its weights, as the old default wrote them |
| `deeptex_compare.py:97-111` | `deeptex_compare.py:165, 176-180` | `save_colour` (the colour branch of `save_sr`); `save_job` refuses a scalar job |
| `deeptex_compare.py:114-138` | `deeptex_compare.py:198-222` | `parser(jobs, default_maps)`, `select()` |
| `deeptex_compare.py:141-223` (`run`) | `deeptex_compare.py:224-300` | the old loop; the writer and the recorded collapse are parameters |

L3 runs old and new on the CPU into the same `--out`: every PNG and manifest
byte-identical (`l3_infer.py`).

## Path dependencies on other arms

- **MatUp baseline.** The `render` job reads `render_lr.png` from the newest
  `runs/matsynth/<m>/matup-output-<m>/<stamp>/` (`matup/baseline/matup_render.py`
  writes it). Every DeepTextures render column in this repo is DeepTextures applied to
  MatUp's render of the LR maps; without a MatUp run the job is skipped
  (`[skip] <m>/render (no MatUp run)`).
- **Checkpoints.** `--ckpt-dir` or `cfg["checkpoint_root"]/<sub_name>` (default
  `/workspace/runs/deeptex/checkpoints`, which the upstream configs do not set - the paper
  and MatSynth arms always pass `--ckpt-dir runs/deeptex/<arm>/checkpoints/esrgan_40k`).
- Consumers: `paper/deeptex_postprocess.py --deeptex-root runs/deeptex/eval/paper`,
  `matsynth/deeptex_drift.py` (via `eval_arms`), `tiling` (`deeptex_sr.py` writes
  `runs/deeptex/tiling/ours`).

## Kept, not fixed

- GPU inference is not bit-reproducible (DEEPTEX-DEVIATIONS 0.9): a rerun moves outputs by
  up to 1 LSB. On the CPU, old and new are byte-identical.
