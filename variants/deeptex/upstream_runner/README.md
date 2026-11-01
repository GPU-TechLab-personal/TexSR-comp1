# deeptex/upstream_runner — upstream's trainer and the record it reads

Shared by every DeepTextures arm that trains with the released `train_esrgan.py`
(paper, matsynth_all / chratio / basecolor).

| file | old file | kind |
|---|---|---|
| `run_upstream_train.py` | `scripts/deeptex/run_upstream_train.py` | moved (L0: AST-identical) |
| `texsr_build_tfrecord.py` | `scripts/deeptex/texsr_build_tfrecord.py` | `main` decomposed; `--repeat` -> `matsynth/recipes.py` |
| `texsr_verify_tfrecord.py` | `scripts/deeptex/texsr_verify_tfrecord.py` | every def identical except `main` (one import line) |

## What changed against the old code (new file:line)

| new | was | change |
|---|---|---|
| `run_upstream_train.py:40-44` | `run_upstream_train.py:40` | bootstrap; `SUBMODULE = paths.DEEPTEX_ROOT` (same DEEPTEX_ROOT override, now from `util/paths.py`) |
| `texsr_build_tfrecord.py:75-87` (`parser`) | `texsr_build_tfrecord.py:80-97` | the shared options, so `matsynth/recipes.py` builds with the same ones |
| `texsr_build_tfrecord.py:90-189` (`build(args, repeats)`) | `texsr_build_tfrecord.py:106-197` | the old body after argument parsing, line for line; returns the meta |
| `texsr_build_tfrecord.py:192-204` (`main`) | `texsr_build_tfrecord.py:82-85, 99-102` | `--hr/--lr` only; every pair packed once (`[1] * len(hr)`, what the old code did without `--repeat`) |
| `texsr_verify_tfrecord.py:37-38, 139` | `texsr_verify_tfrecord.py:133-135` | `paths.use_deeptex()` instead of the function-local `sys.path.insert(DEEPTEX_ROOT default)`; still lazy |

The record, its meta (`"repeat": [1]` included) and the verify report are the old
ones: L2 rebuilds textures_db with both and compares every record (`l2_data.py tdb`).

## Path dependencies on other arms

- `texsr_build_tfrecord.build()` is called by `matsynth/recipes.py`; `_bytes_feature`
  and `sha256` are imported by `render_loss/build/texsr_build_grouped_tfrecord.py`.
- Inputs: `runs/deeptex/datasets/textures_db/{HR,LRx4}` (from `paper/texsr_make_pairs.py`)
  and `runs/deeptex/datasets/matsynth/<map>/{HR,LRx4}` (from `matsynth/matsynth_make_pairs.py`).
- `run_upstream_train.py` runs `external/DeepTextures/train_esrgan.py` with CWD = the arm's
  root (`runs/deeptex/<arm>`), so `./checkpoints` land there; the configs carry absolute
  record paths for that reason.

## Kept, not fixed

- `*.meta.json["sha256"]` is the FILE hash, and the file bytes are not reproducible: the
  protobuf runtime serialises an Example's three-key feature map in no fixed order. Same
  content, other hash (see `paper/README.md`). Use the record content (`l2_records.py`).
