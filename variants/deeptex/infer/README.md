# variants/deeptex/infer/

Running a trained DeepTextures checkpoint: a directory-in / PNGs-out super-resolver
that replaces upstream's unusable `test.py`, and the shared core that runs a
checkpoint over the 12 evaluation materials' LR inputs so the comparison layer can
add a DeepTextures column. Used by every DeepTextures arm's evaluation.

## Contents

| file | what it does |
|---|---|
| `deeptex_sr.py` | deeptex container: restore a generator (`--ckpt-dir` / `--ckpt`, or `cfg sub_name`), check the restore changed the weights, `--self-test`, super-resolve every image of `--in-dir` into `--out-dir` (optional LR-space tiling with overlap, `--make-lr bicubic`, `--save-bicubic`); writes `sr_manifest.json`. Also the library (`check_checkpoint`, `super_resolve`, `die`) the other inference scripts call |
| `deeptex_compare.py` | deeptex container: the shared core - for each material under `--root`, baseColor (from `SVBRDF/`) and render (MatUp's `render_lr.png`) through the checkpoint into `<out>/<m>/{baseColor,render}.png` + `sr_manifest.json`; `parser()`, `select()`, `run()`, `load_as_rgb01()`, `save_colour()` are reused by `matsynth/eval_arms.py` |

## How it works

`deeptex_compare.run(args, jobs, save, collapse)`:

1. Build `RRDB_Model(None, ...)` (dynamic size), resolve the checkpoint
   (`--ckpt-dir`, else `cfg["checkpoint_root"]/<sub_name>`, default root
   `/workspace/runs/deeptex/checkpoints`), restore it and fail if no weight changed.
2. For every `runs/matsynth/<m>/` with an `SVBRDF/` (or `--material`): find the newest
   MatUp run (`matup/baseline/matup_outputs.newest_run`, or `--run <stamp>`).
3. Per job: `load_as_rgb01` (cv2 read, `variants/util/map_encode.to_rgb8` to 8-bit
   RGB - any of the five map encodings or the 16-bit render), `deeptex_sr.super_resolve`,
   write with `save(path, sr, kind)` (`save_colour`: rint, RGB -> BGR, PNG).
4. Write `<out>/<m>/sr_manifest.json`: checkpoint prefix and step, `sub_name`, scale,
   the MatUp run used, `scalar_collapse` / `collapse_weights`, and per output its source,
   encoding, LR and SR shapes (and `channel_spread` for 1-channel maps).

`deeptex_compare.py` itself runs the two colour jobs (`--maps baseColor,render`) and
never collapses; `matsynth/eval_arms.py` runs the same `run()` with all six jobs and a
writer that collapses the 1-channel maps.

Inputs and consumers:

- **MatUp baseline**: the `render` job reads `render_lr.png` from the newest
  `runs/matsynth/<m>/matup-output-<m>/<stamp>/` (`variants/matup/baseline/matup_render.py`
  writes it). Every DeepTextures render column in this repo is DeepTextures applied to
  MatUp's render of the LR maps; without a MatUp run the job is skipped
  (`[skip] <m>/render (no MatUp run)`).
- **Checkpoints**: the upstream configs set no `checkpoint_root`, so the paper and
  MatSynth arms always pass `--ckpt-dir runs/deeptex/<arm>/checkpoints/esrgan_40k`.
- **Consumers**: `paper/run.ps1 -Stage eval` (-> `runs/deeptex/eval/<tag>`, read by
  `paper/deeptex_postprocess.py --deeptex-root`), `paper/run.ps1 -Stage smoke`
  (`deeptex_sr.py --self-test`), `matsynth/eval_arms.py` and `matsynth/deeptex_drift.py`
  (import the core), the tiling study (`deeptex_sr.py` -> `runs/deeptex/tiling/ours`).

## Run

```
docker compose run --rm deeptex python /workspace/variants/deeptex/infer/deeptex_compare.py \
    --cfg /workspace/variants/deeptex/paper/esrgan_paper.yaml \
    --ckpt-dir /workspace/runs/deeptex/paper/checkpoints/esrgan_40k \
    --out /workspace/runs/deeptex/eval/paper

docker compose run --rm deeptex python /workspace/variants/deeptex/infer/deeptex_sr.py \
    --cfg /workspace/variants/deeptex/paper/esrgan_paper.yaml \
    --ckpt-dir /workspace/runs/deeptex/paper/checkpoints/esrgan_40k \
    --in-dir <dir> --out-dir <dir> [--tile-size 128 --overlap 16] [--self-test]
```

Normally both are run by `variants/deeptex/paper/run.ps1` (`-Stage eval`, `-Stage smoke`).

Exit codes (`deeptex_sr.die`, also used by `deeptex_compare.py`): 2 usage / no inputs /
`--maps` matched nothing, 3 checkpoint missing or incomplete, restore changed nothing,
self-test failed, 4 non-finite output, 10 OOM (lower `--tile-size`), 11 GPU lost. Each
prints a `[fatal]` line first.

## Known behaviour and defects

- GPU inference is not bit-reproducible (DEEPTEX-DEVIATIONS 0.9): a rerun moves outputs
  by up to 1 LSB. On the CPU, reruns are byte-identical.
- Every `sr_manifest.json` from `deeptex_compare.py` records `"scalar_collapse": "mean"`
  and its weights although no map is collapsed there (`RECORDED_COLLAPSE`).
- `deeptex_compare.py`'s docstring points to `scripts/deeptex/deeptex_compare.py:7-64`
  for its scope discussion; that file no longer exists - the discussion is in
  `docs/DEEPTEX-DEVIATIONS.md` 0.6. `deeptex_sr.py` cites `scripts/common/texsr_metrics.py`
  (now `variants/util/texsr_metrics.py`).
