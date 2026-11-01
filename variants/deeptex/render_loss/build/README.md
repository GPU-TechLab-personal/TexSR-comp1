# variants/deeptex/render_loss/build/

Builds and verifies the grouped TFRecord the render-loss arms train on: one
Example per material holding all ten PNG strings (five maps x HR/LR), so a
training batch can be rendered (`render_loss/grouped_dataset.py`,
`render_loss/render_tf.py`). The paper-path record holds one map per Example
and crops each on its own, so no batch from it is renderable.

## Contents

| file | what it does |
|---|---|
| `texsr_build_grouped_tfrecord.py` | packs the MatSynth pairs into `grouped_dataset.SCHEMA` Examples (bytes copied, never re-encoded), in a seeded shuffled order, and writes `<out>.meta.json`; stage `prep-grouped`, step 1 |
| `texsr_verify_grouped.py` | proves the record is exactly those pairs (meta, keys, names, order, PNG headers, decode, byte comparison, finite GT decode) and writes `verify_report.json` with PASS / PARTIAL / FAIL; step 2 |

## How it works

Input: the MatSynth pair root `runs/deeptex/datasets/matsynth`
(`variants/deeptex/matsynth/matsynth_make_pairs.py`, the same folders the
`matsynth/recipes.py` records are packed from):

* `<map>/{HR,LRx4}/<safe>__<map>.png` for baseColor, normal, roughness,
  metallic, height;
* `_meta/<safe>.json`, written last per material, so its `safe` list is the
  material list; the ten folder listings must agree with it (split with
  `rsplit("__", 1)`, every file ending in exactly `__<map>.png`), or the build
  stops before the writer opens (exit 4);
* `pairs_manifest.json`, whose sha256 goes into the meta.

Build: names sorted, then shuffled with `random.Random(seed)` (default 0). The
render arms train with `--shuffle off`, so this order is the curriculum R0a, R0b
and R1 share. Before writing, the old `<out>.meta.json`, `verify_report.json`
and any `.partial` are deleted, so a crashed build leaves a record the trainer
refuses. One record per folder is enforced (the trainer reads
`dirname(record)/verify_report.json`). It imports `_bytes_feature` / `sha256`
from `variants/deeptex/upstream_runner/texsr_build_tfrecord.py`.

Output `runs/deeptex/datasets/matsynth_grouped/`:

| file | read by |
|---|---|
| `matsynth_grouped_lrx4.tfrecord` | `render_loss/hooks.py`, `render_loss/gates/render_gate.py`, `render_loss/gates/render_calibrate.py` (via `grouped_dataset.load_grouped_dataset`) |
| `matsynth_grouped_lrx4.tfrecord.meta.json` | `record_count`, `hr_size`, `lr_size`: the loader; `schema`, `sha256`, `bytes`: the trainer's dataset checks; `seed`, `limit`, `root`, `order_sha256`: the verifier |
| `verify_report.json` | the trainer: verdict must be PASS and its `meta` block must equal the record's own meta |

Verify: 2,904 materials; every string byte-compared with its pair file (29,040
comparisons). PASS needs a complete check (`--sample 0`) of a full build (no
`--limit`); a sampled check or a `--limit` record gives PARTIAL, which the
trainer refuses. It also reports, without failing, grey-map channel equality and
per material the fraction of GT normal texels the render loss masks
(`normal_invalid`, from `render_tf.decode_maps`). The GPU is hidden.

## Run

Through `variants\deeptex\render_loss\run.ps1 -Stage prep-grouped` (an existing
record is only re-verified unless `-Force`). Directly, in the deeptex container:

    python /workspace/variants/deeptex/render_loss/build/texsr_build_grouped_tfrecord.py [--force]
    python /workspace/variants/deeptex/render_loss/build/texsr_verify_grouped.py

Defaults: `--root /workspace/runs/deeptex/datasets/matsynth`, `--out` / `--tfrecord`
`/workspace/runs/deeptex/datasets/matsynth_grouped/matsynth_grouped_lrx4.tfrecord`.
Builder flags `--seed`, `--limit N` (a prefix of the full order, for quick tests),
`--sample` (records decode-checked after writing, default 16); verifier
`--sample N` (per-string checks on the first N records only; default 0 = all).

Exit codes. Builder: 0 ok; 2 bad arguments, `--out` under `external/` or beside
another record; 3 `--out` exists (pass `--force`); 4 pair folders inconsistent;
5 the written record failed its decode check (no meta written). Verifier: 0 PASS,
1 FAIL, 2 bad arguments, 3 PARTIAL.

## Known behaviour and defects

* The record's file sha256 (in the meta, and checked by the verifier against the
  file it reads) is not reproducible across rebuilds: protobuf map order. A
  rebuilt record must be verified again; the content and `order_sha256` are what
  stay fixed.
* The meta's `"builder"` is `os.path.basename(__file__)`,
  `texsr_build_grouped_tfrecord.py`.
* Docstrings refer to the trainer as `deeptex_train.py`; that is now
  `paper_loop/main.py --experiment render` with `render_loss/hooks.py`.
