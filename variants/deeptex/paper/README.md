# variants/deeptex/paper/

The paper arm: DeepTextures as released - upstream's `train_esrgan.py`, unmodified,
on the released `external/DeepTextures/data/textures_db` - scored next to MatUp on
the 12 MatSynth evaluation materials. This folder holds the textures_db data prep,
the arm's config, the orchestrator, and the hooks that add the DeepTextures column
and sheet to the MatUp comparison. Training itself is
`upstream_runner/run_upstream_train.py`; inference is `infer/deeptex_compare.py`.

## Contents

| file | what it does |
|---|---|
| `run.ps1` | orchestrator: stages `preflight`, `prep`, `smoke`, `paper`, `eval`, `all`; also trains the MatSynth arms (`-PaperConfig` / `-PaperRoot`) |
| `esrgan_paper.yaml` | upstream `configs/esrgan.yaml @ e4ee562` with one line changed (`train_dataset.path`, absolute) |
| `texsr_normalize.py` | stage [A], dataprep container: any image dir -> 8-bit, 3-channel PNGs with sides a multiple of 4 and >= 128, every conversion counted in `normalize_manifest.json` |
| `texsr_make_pairs.py` | stage [B], deeptex container: tiles HR and derives each LRx4 tile with upstream's `imresize_np`; holdout split by source image; writes `pairs_manifest.json` |
| `deeptex_postprocess.py` | `matup/baseline/postprocess.py` plus the `DeeptexColumn` hook: a fifth column (`deeptex`) on baseColor and render, with its sharpness metrics and console notes |
| `deeptex_montage.py` | `matup/baseline/montage.py` plus the `DeeptexSheet` hook: the separate 5x2 `montage_deeptex.png` and its `--collect` copy `<material>_deeptex.png` |

## How it works

```
external/DeepTextures/data/textures_db  (300 files: png/jpg, RGBA/palette/16-bit, 64^2..2048^2)
  -> texsr_normalize.py          runs/deeptex/datasets/textures_db/normalized/  + normalize_manifest.json
  -> texsr_make_pairs.py         .../textures_db/{HR,LRx4,HR_val,LRx4_val}/      + pairs_manifest.json
  -> upstream_runner/texsr_build_tfrecord.py   .../textures_db_lrx4.tfrecord (+ .meta.json)
  -> upstream_runner/texsr_verify_tfrecord.py  .../verify_report.json
  -> upstream_runner/run_upstream_train.py     (CWD runs/deeptex/paper)
                                 runs/deeptex/paper/checkpoints/esrgan_40k/
  -> infer/deeptex_compare.py    runs/deeptex/eval/paper/<m>/{baseColor,render}.png + sr_manifest.json
  -> deeptex_postprocess.py      runs/matsynth/<m>/compare/<stamp>/  (fifth column + metrics.json)
  -> deeptex_montage.py          compare/<stamp>/montage{,_deeptex}.png; --collect -> runs/montages/
```

Stages of `run.ps1` (each step is a `docker compose run` of `docker/compose.yaml`,
logged to `runs/_logs/<stamp>/<step>.log`; helpers from `variants/util/lib.ps1`):

| stage | runs |
|---|---|
| `preflight` | `Invoke-DeeptexPreflight`: no stray containers, >= 6000 MiB VRAM free (warns below 8000), >= 15 GB disk. Runs before every stage unless `-SkipPreflight` |
| `prep` | `texsr_normalize.py` (dataprep) -> `texsr_make_pairs.py --crop-size 480 --step 240 --lr-kernel bicubic --jobs 8` -> `texsr_build_tfrecord.py --overwrite` -> `texsr_verify_tfrecord.py`; a FAIL stops before training |
| `smoke` | 20 steps of the **fixed** trainer (`paper_loop/main.py --experiment fixed --stage psnr`, config `variants/deeptex/fixed/psnr_texsr.yaml`, into `runs/deeptex/checkpoints/smoke_psnr`), then `infer/deeptex_sr.py --self-test` on `acg_concrete_026` into `runs/deeptex/eval/_smoke`. Proves CUDA, the VGG19 weights, checkpoint writes and the train -> eval handoff in about four minutes |
| `paper` | `run_upstream_train.py --cfg_path <PaperConfig>` with CWD `/workspace/<PaperRoot>`; then a `python -c` probe that loads `checkpoints/<sub_name>` and requires its `step` tensor >= `niter` (prints `[ckpt] ... step=N`) |
| `eval` | `infer/deeptex_compare.py --root runs/matsynth --out runs/deeptex/eval/<EvalTag>` -> `deeptex_postprocess.py --all --metrics --deeptex-root <out>` -> `deeptex_montage.py --all --collect /workspace/runs/montages` |
| `all` (default) | `prep`, `smoke`, `paper`, `eval` |

The two hooks are objects passed as `extra` to `matup/baseline/postprocess.run()` and
`matup/baseline/montage.run()`; without `--deeptex-root` `deeptex_postprocess.py`
writes exactly what `matup/baseline/postprocess.py` writes. The column is cropped into
MatUp's valid region with `util/report/images.load_fit()`, which the MatSynth and
render-loss reports use too. Only baseColor and render get a
DeepTextures column here; the all-maps comparison is `variants/deeptex/matsynth/`.

What it reads from other folders:

- **MatUp baseline** (`variants/matup/baseline/`): `runs/matsynth/<m>/{SVBRDF,GT}` and
  the newest `matup-output-<m>/<stamp>/` (`*_opt.png`, `height_opt.png`,
  `render_{lr,bilinear,opt,gt}.png`). The `render` input to DeepTextures is MatUp's
  `render_lr.png` (newest run, `matup/baseline/matup_outputs.newest_run`): every DeepTextures
  render column, here and in `matsynth/` and `render_loss/eval/`, is DeepTextures
  applied to MatUp's render of the LR maps.
- `eval` writes into the MatUp tree (`runs/matsynth/<m>/compare/<stamp>/`) and rewrites
  the committed `runs/montages/`.

### The config

`esrgan_paper.yaml` differs from upstream in its 14-line comment header plus a blank
line, and exactly one line:

```diff
-    path: './data/textures_db_normalized/textures_db_lrx4.tfrecord'
+    path: '/workspace/runs/deeptex/datasets/textures_db/textures_db_lrx4.tfrecord'
```

Upstream's path resolves against the CWD, and the arm runs with CWD
`runs/deeptex/paper` so its `./checkpoints` and `./logs` stay outside the submodule.
Check (the submodule checkout has CRLF line endings):

```
diff --strip-trailing-cr <(sed '1,15d' variants/deeptex/paper/esrgan_paper.yaml) external/DeepTextures/configs/esrgan.yaml
```

`run.ps1` reads `niter`, `save_steps` and `sub_name` from the config with line
regexes (`^niter:\s*(\d+)`, `^sub_name:\s*'(.+)'`), so those keys must stay top-level,
one per line, `sub_name` single-quoted. Everything else (batch_size 8, sub_name
`esrgan_40k`, the unresolvable `pretrain_name`, `lr_steps` that never fire) is the
released baseline and is not changed.

## Run

From the repo root (`powershell -NoProfile -ExecutionPolicy Bypass -File variants\deeptex\paper\run.ps1 ...`):

```
variants\deeptex\paper\run.ps1 -Stage prep
variants\deeptex\paper\run.ps1 -Stage smoke
variants\deeptex\paper\run.ps1 -Stage paper -KeepAllCheckpoints
variants\deeptex\paper\run.ps1 -Stage eval -EvalTag paper

# A MatSynth arm's training (then variants\deeptex\matsynth\run.ps1 -Stage eval-arms):
variants\deeptex\paper\run.ps1 -Stage paper -PaperConfig variants/deeptex/matsynth/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all
```

Parameters that matter here:

| parameter | default | effect |
|---|---|---|
| `-Dataset` / `-SourceDir` | `textures_db` / `external/DeepTextures/data/textures_db` | prep input and `runs/deeptex/datasets/<Dataset>/` |
| `-CropSize` / `-Step` / `-LrKernel` | 480 / 240 / bicubic | `texsr_make_pairs.py` tiling |
| `-PaperConfig` / `-PaperRoot` | `variants/deeptex/paper/esrgan_paper.yaml` / `runs/deeptex/paper` | trainer config and CWD; each arm needs its own root |
| `-KeepAllCheckpoints` | off | a host job (`Start-CheckpointKeeper`) copies each checkpoint to `<PaperRoot>/checkpoints_all/<sub_name>/` as it appears (upstream keeps only 3) |
| `-MaxSteps N` | 0 | writes `<PaperRoot>/esrgan_paper_maxsteps.yaml` with `niter: N`, `save_steps: N/2` and trains on that |
| `-Force` / `-DeleteCheckpoints` | off | prep: overwrite; paper: `-Force` alone resumes into the existing checkpoints, with `-DeleteCheckpoints` moves them to `<ckpt>.bak-<stamp>` first |
| `-Config` / `-EvalTag` / `-CkptDir` | paper config / `<Dataset>` / derived | `eval`: with the paper config and no `-CkptDir`, the checkpoint is `<PaperRoot>/checkpoints/<sub_name>` |

The param block is shared by all four DeepTextures `run.ps1` files, so parameters of
other files' stages (`-Recipe`, `-Arm`, `-Gate`, `-ExtraArms`, ...) are accepted and
ignored here.

## Known behaviour and defects

- The check written in `esrgan_paper.yaml:5-6` (`diff <(sed '1,12d' ...)`) is stale:
  the header is 15 lines, so it leaves three comment lines in, and without
  `--strip-trailing-cr` every line differs. Use the command above.
- `texsr_make_pairs.py` fills `pairs_manifest.json["tiles_by_source"]` in
  `imap_unordered` completion order, so two runs write the same content with keys in a
  different order. Compare it as JSON, not as bytes.
- The TFRecord's `sha256` in `*.meta.json` is a file hash, not a content identity: the
  protobuf runtime serialises an Example's feature map in no fixed order, so rebuilding
  the same record gives another file hash (committed `8b08886c...`, 2965 records;
  rebuilds gave `8116ad85...` and `25d46f63...` with every record identical).
- `paper` detects failure from the trainer's output (`Get-FailureKind`: cpu-fallback,
  gpu-lost, oom, data) and from `[metrics] step=<niter>`; a run that printed `[metrics]`
  but saved no checkpoint at `niter` fails at the `[ckpt]` probe.
- Some docstrings cite `scripts/...` / `configs/...` paths and line numbers that no
  longer exist in the repo.
