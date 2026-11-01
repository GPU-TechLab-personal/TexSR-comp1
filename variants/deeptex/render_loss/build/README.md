# deeptex/render_loss/build — the grouped record of the render-loss arms

One Example per material holding all ten PNG strings (five maps x HR/LR), so a
training batch can be rendered (`render_loss/grouped_dataset.py`, `render_tf.py`).

| file | old file | kind |
|---|---|---|
| `texsr_build_grouped_tfrecord.py` | `scripts/deeptex/texsr_build_grouped_tfrecord.py` | moved (L0: AST-identical; `REPO` allowed) |
| `texsr_verify_grouped.py` | `scripts/deeptex/texsr_verify_grouped.py` | moved (L0: AST-identical) |

## What changed against the old code (new file:line)

| new | was | change |
|---|---|---|
| `texsr_build_grouped_tfrecord.py:83-89` | `texsr_build_grouped_tfrecord.py:83-89` | bootstrap; `grouped_dataset` from `render_loss/`, `_bytes_feature`/`sha256` from `upstream_runner/texsr_build_tfrecord.py` |
| `texsr_build_grouped_tfrecord.py:94` | `texsr_build_grouped_tfrecord.py:94` | `REPO = paths.REPO` (used by `under_external`) |
| `texsr_verify_grouped.py:78-85` | `texsr_verify_grouped.py:78-85` | bootstrap; `grouped_dataset`, `render_tf`, the builder by package |

The meta's `"builder"` is `os.path.basename(__file__)` and stays
`texsr_build_grouped_tfrecord.py`. L2 rebuilds the record with old and new: identical
record by record, identical `order_sha256`, verify reports identical
(`l2_data.py grouped`).

## Path dependencies on other arms

- Input: the MatSynth pair root `runs/deeptex/datasets/matsynth` (`matsynth/matsynth_make_pairs.py`):
  `<map>/{HR,LRx4}/<safe>__<map>.png`, `_meta/<safe>.json` (the material list) and
  `pairs_manifest.json` (its sha256 goes into the meta) - the same folders the
  `matsynth/recipes.py` records are packed from.
- Output `runs/deeptex/datasets/matsynth_grouped/` (one record per folder: the trainer reads
  one `verify_report.json` per folder). Read by the render-loss trainer hooks and gates
  (not in this folder).

## Kept, not fixed

- (Fixed when the old tree was deleted: `texsr_build_grouped_tfrecord.py:471`'s `next:` line
  printed the old `scripts/deeptex/` path; it now names this folder's verifier.)
- The record's file sha256 (in the meta and checked by the verifier against the file it
  reads) is not reproducible across rebuilds: protobuf map order. A rebuilt record must be
  verified again; the content and `order_sha256` are what stay fixed.
