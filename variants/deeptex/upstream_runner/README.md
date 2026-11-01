# variants/deeptex/upstream_runner/

Runs the submodule's own trainer, `external/DeepTextures/train_esrgan.py`, unmodified,
and builds and checks the TFRecords it reads. Shared by every DeepTextures arm that
trains with the released trainer (paper, matsynth_all / chratio / basecolor); the
record builder is also used by `matsynth/recipes.py` and, in part, by
`render_loss/build/`.

## Contents

| file | what it does |
|---|---|
| `run_upstream_train.py` | deeptex container: execs `train_esrgan.py --cfg_path <cfg> --gpu <n>` with `PYTHONPATH` = the submodule and the caller's CWD; splits its `\r` progress bar into `[metrics]` lines; never touches the training process itself |
| `texsr_build_tfrecord.py` | stage [C], deeptex container: packs one or more `--hr/--lr` pair folders into ONE record with upstream's feature schema (`image/img_name`, `image/hr_encoded`, `image/lr_encoded`), one seeded shuffle, decode-checks a sample, writes `<out>.meta.json`; `parser()` and `build(args, repeats)` are the library `matsynth/recipes.py` calls |
| `texsr_verify_tfrecord.py` | stage [D], deeptex container: one full pass decoding with `channels=0` - unique non-empty `img_name`, uint8, 3 channels, exact x4 shapes, HR >= gt_size and LR >= gt_size/scale, record count >= batch size, x4 relation recomputed with `imresize_np` on `--align-sample` records, count and sha256 against the meta; writes `verify_report.json` beside the record |

## How it works

```
<pairs>/{HR,LRx4}  (paper/texsr_make_pairs.py: runs/deeptex/datasets/textures_db/
                    matsynth/matsynth_make_pairs.py: runs/deeptex/datasets/matsynth/<map>/)
  -> texsr_build_tfrecord.py  (or matsynth/recipes.py -> build())
         <record>.tfrecord + <record>.tfrecord.meta.json
           meta: record_count, seed, scale, sha256, bytes, hr_dir, lr_dir, repeat, pairs_manifest, schema
  -> texsr_verify_tfrecord.py
         verify_report.json (verdict PASS / FAIL, failures, warnings, meta)
  -> run_upstream_train.py   CWD = runs/deeptex/<arm>
         train_esrgan.py reads cfg train_dataset.path (absolute in our configs)
         and writes ./checkpoints/<sub_name>/ and ./logs/ under that CWD
```

Who calls it: `paper/run.ps1 -Stage prep` (build + verify for textures_db),
`-Stage paper` (`run_upstream_train.py`, for the paper arm and, with
`-PaperConfig/-PaperRoot`, the MatSynth arms), `matsynth/run.ps1 -Stage prep-matsynth`
(verify; the build goes through `recipes.py`). `render_loss/build/texsr_build_grouped_tfrecord.py`
imports `_bytes_feature` and `sha256` from `texsr_build_tfrecord.py`.

The CWD is the arm's root so that `./checkpoints` and `./logs` land outside the
submodule; that is why every config carries an absolute record path.

### Output contract of `run_upstream_train.py`

`run.ps1` parses these lines; keep them byte-for-byte:

- `[metrics] step=<n> niter=<N> loss_G=<f> loss_D=<f> lr_G=<f> steps_per_sec=<f>` -
  the first frame, every `--log-every` steps (default 100) and the last.
- On resume at or past `niter` (`load ckpt from ... at step N` and no progress line):
  `[*] already complete: ...` and one `[metrics] step=N niter=N ... nan ...` line.
- Zero training steps without a resume (empty record): `[fatal] upstream trainer
  produced no training steps at all ...`, exit 5. Otherwise the trainer's exit code.
- Every other trainer line is passed through unchanged.

`texsr_verify_tfrecord.py` prints `[warn]` / `[FAIL]` lines and `verdict: PASS|FAIL`
and exits 1 on FAIL.

## Run

Normally through `variants/deeptex/paper/run.ps1` (`-Stage prep`, `-Stage paper`) and
`variants/deeptex/matsynth/run.ps1 -Stage prep-matsynth`. Directly:

```
docker compose run --rm deeptex python /workspace/variants/deeptex/upstream_runner/texsr_build_tfrecord.py \
    --hr /workspace/runs/deeptex/datasets/textures_db/HR --lr /workspace/runs/deeptex/datasets/textures_db/LRx4 \
    --out /workspace/runs/deeptex/datasets/textures_db/textures_db_lrx4.tfrecord \
    --pairs-manifest /workspace/runs/deeptex/datasets/textures_db/pairs_manifest.json --overwrite

docker compose run --rm deeptex python /workspace/variants/deeptex/upstream_runner/texsr_verify_tfrecord.py \
    --tfrecord /workspace/runs/deeptex/datasets/textures_db/textures_db_lrx4.tfrecord

docker compose run --rm -w /workspace/runs/deeptex/paper deeptex \
    python /workspace/variants/deeptex/upstream_runner/run_upstream_train.py \
    --cfg_path /workspace/variants/deeptex/paper/esrgan_paper.yaml --gpu 0 --log-every 100
```

`--hr/--lr` may be repeated (paired in order); every pair is packed once. The
per-map repeat counts of the MatSynth mixes are `matsynth/recipes.py`.

## Known behaviour and defects

- `*.meta.json["sha256"]` is the FILE hash, and the file bytes are not reproducible:
  the protobuf runtime serialises an Example's three-key feature map in no fixed order.
  The same content rebuilt gives another hash (see `paper/README.md`); compare records
  by content. The verify step checks the hash only against the meta written with the
  same file.
- The meta records `"repeat": [1, ...]` for a plain build; `recipes.py` records its
  recipe's counts.
- Upstream never calls `tf.random.set_seed`, so a training run is not deterministic,
  and it has no validation loop: evaluate the periodic checkpoints (every
  `save_steps`) from a separate process with `infer/`.
- Upstream keeps only the last three checkpoints (`max_to_keep=3`); use
  `paper/run.ps1 -KeepAllCheckpoints` to keep every one.
- `run_upstream_train.py`'s docstring names `scripts/deeptex/deeptex_sr.py` and
  `run_deeptex.ps1`; they are now `infer/deeptex_sr.py` and `paper/run.ps1`.
