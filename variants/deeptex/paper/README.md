# deeptex/paper — the paper arm's data prep and its DeepTextures column

The released DeepTextures trainer on the released `data/textures_db`. Training itself
is `upstream_runner/run_upstream_train.py` (upstream's `train_esrgan.py`, verbatim);
this folder holds what feeds it and what puts its result next to MatUp.

| file | old file | kind |
|---|---|---|
| `texsr_normalize.py` | `scripts/deeptex/texsr_normalize.py` | moved (L0: AST-identical) |
| `texsr_make_pairs.py` | `scripts/deeptex/texsr_make_pairs.py` | moved; `SUBMODULE = paths.use_deeptex()` at :53 |
| `esrgan_paper.yaml` | `configs/deeptex/esrgan_paper.yaml` | byte-identical copy |
| `deeptex_postprocess.py` | `scripts/compare/matup_postprocess.py` `--deeptex-root` | split out (decomposed) |
| `deeptex_montage.py` | `scripts/compare/matup_montage.py` `montage_deeptex.png` | split out (decomposed) |

## The config: one line against upstream

`esrgan_paper.yaml` is `external/DeepTextures/configs/esrgan.yaml @ e4ee562` with a
12-line comment header and exactly one changed line:

```diff
-    path: './data/textures_db_normalized/textures_db_lrx4.tfrecord'
+    path: '/workspace/runs/deeptex/datasets/textures_db/textures_db_lrx4.tfrecord'
```

(upstream's path resolves against the CWD, and the paper arm runs with CWD
`runs/deeptex/paper` so its `./checkpoints` stay outside the submodule). Check:
`diff --strip-trailing-cr <(sed '1,15d' variants/deeptex/paper/esrgan_paper.yaml) external/DeepTextures/configs/esrgan.yaml`
— the submodule checkout has CRLF line endings. The file's own header still says
`configs/deeptex/esrgan_paper.yaml` and `sed '1,12d'`: it is copied byte for byte
because the ps1 files read `sub_name` from it with a line regex.

## What changed against the old code (new file:line)

| new | was | change |
|---|---|---|
| `texsr_make_pairs.py:51-53` | `texsr_make_pairs.py:51-52` | bootstrap + `paths.use_deeptex()` instead of the DEEPTEX_ROOT default and `sys.path.insert` |
| `deeptex_postprocess.py:32-34` | `matup_postprocess.py:80-82` | `DEEPTEX_COLUMN`, `DEEPTEX_MAPS`, `ORDER_WITH_DEEPTEX` |
| `deeptex_postprocess.py:37-79` (`DeeptexColumn`) | `matup_postprocess.py:303, 308-313, 340-345, 365-370, 400-402, 408-409` | the column as a hook that `variants/compare/matup_postprocess.process(extra=)` calls |
| `deeptex_postprocess.py:82-88` | `matup_postprocess.py:428-430` | `--deeptex-root`; without it the run is `compare/matup_postprocess.py` |
| `deeptex_montage.py:35-41` | `matup_montage.py:55-61` | the 5x2 grid and its 160 px gutter |
| `deeptex_montage.py:44-81` (`DeeptexSheet`) | `matup_montage.py:173-185, 201, 208-212, 226-237, 255` | the second sheet, its `--collect` copy and the note above the summary table |

`deeptex_column()` (the crop of an SR output into MatUp's valid region) is NOT here:
the MatSynth and render-loss reports use it too, so it stays in
`variants/compare/matup_postprocess.py`.

## Path dependencies on other arms

- **MatUp baseline.** `deeptex_postprocess.py` reads `runs/matsynth/<m>/{SVBRDF,GT}` and the
  newest `matup-output-<m>/<stamp>/` (`*_opt.png`, `height_opt.png` from
  `matup/baseline/matup_height.py`, `render_{lr,bilinear,opt,gt}.png` from
  `matup/baseline/matup_render.py`), and writes into that MatUp tree:
  `runs/matsynth/<m>/compare/<stamp>/`. `deeptex_montage.py --collect` rewrites the
  committed `runs/montages/`.
- **Inference.** The column is `runs/deeptex/eval/paper/<m>/{baseColor,render}.png`
  from `infer/deeptex_compare.py`. Its `render` input is MatUp's `render_lr.png`
  (the newest run, `compare/matup_runs.newest_run`) - every DeepTextures render
  column, here and in `matsynth/` and `render_loss/eval/`, is DeepTextures applied to
  MatUp's render of the LR maps.
- **Training data.** `texsr_normalize.py` reads `external/DeepTextures/data/textures_db`;
  `upstream_runner/texsr_build_tfrecord.py` packs `runs/deeptex/datasets/textures_db/{HR,LRx4}`.

## Kept, not fixed

- `esrgan_paper.yaml:5-6`'s own check (`diff <(sed '1,12d' ...) external/.../esrgan.yaml`)
  is stale: the header is 15 lines, so it leaves three comment lines in, and without
  `--strip-trailing-cr` every line differs (CRLF submodule). The command above is the
  one that shows the single line.
- `texsr_make_pairs.py:204-206` fills `pairs_manifest.json["tiles_by_source"]` in
  `imap_unordered` completion order, so two runs of the same code write the same
  content with keys in a different order (L2 compares it as JSON for that reason).
- The TFRecord's `sha256` in `*.meta.json` is not a content identity: the protobuf
  runtime serialises an Example's feature map in no fixed order, so rebuilding the
  same record gives another file hash (committed `8b08886c...`; two rebuilds in L2
  gave `8116ad85...` and `25d46f63...` with every record identical).
- Docstrings of moved files cite line numbers of the OLD files (`scripts/...`).

## run.ps1

Stages `prep`, `smoke`, `paper`, `eval`, `all` (the default, as before:
prep → smoke → paper → eval), plus `preflight`. Generated from `scripts/deeptex/run_deeptex.ps1`: each stage function below is the old one verbatim except the script/config paths (old line ranges in the file header). Shared helpers come from `variants/util/lib.ps1`. L5 (`variants/tests/l5_deeptex.ps1`) runs every stage through both and compares the docker argv, the console transcript and the files created.
Kept as they were:

- `smoke` trains through the **fixed** trainer (`paper_loop/main.py --experiment fixed
  --stage psnr`), exactly as the old smoke stage ran `deeptex_train.py --stage psnr`.
- The MatSynth arms train through `-Stage paper` with `-PaperConfig
  variants/deeptex/matsynth/<cfg>.yaml -PaperRoot runs/deeptex/<arm>`; their
  evaluation is `matsynth/run.ps1 -Stage eval-arms`.

Changed: `-KeepAllCheckpoints` calls `Start-CheckpointKeeper` / `Stop-CheckpointKeeper`
from `lib.ps1` instead of its inlined copy of the same `Start-Job` loop (L5
`paper-keepall`: same console lines, same files).
