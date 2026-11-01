# variants/deeptex/matsynth/

The MatSynth arms: the released DeepTextures trainer, unmodified, trained on MatSynth
train-split pairs instead of textures_db and scored on every map (baseColor, normal,
roughness, metallic, height) plus the render. Three training mixes - `all` (the
paper's), `chratio` (ablation) and `basecolor` (control) - each compared with the
textures_db checkpoint and with MatUp. Training runs through
`variants/deeptex/paper/run.ps1 -Stage paper`; this folder holds the pairs, the
recipes, the all-maps evaluation and the reports.

## Contents

| file | what it does |
|---|---|
| `run.ps1` | orchestrator: stages `preflight`, `prep-matsynth`, `eval-arms` |
| `matsynth_make_pairs.py` | dataprep container: 512^2 HR / 128^2 LRx4 pairs per map from MatSynth's `train-*.parquet`, the evaluation materials and their blends excluded; `--check-eval` rebuilds the 12 eval materials and requires pixel identity with `runs/matsynth` |
| `recipes.py` | the three recipes (how many times each map folder is packed) and the one TFRecord per recipe, through `upstream_runner/texsr_build_tfrecord.build()` |
| `esrgan_matsynth.yaml` | config of the `all` arm |
| `esrgan_matsynth_chratio.yaml` | config of the `chratio` arm |
| `esrgan_matsynth_basecolor.yaml` | config of the `basecolor` arm |
| `eval_arms.py` | all six inputs (five maps + MatUp's `render_lr.png`) through a checkpoint, via the `infer/deeptex_compare.py` core; 1-channel maps collapsed 3 -> 1 (`--scalar-collapse mean\|luma\|paper`) |
| `deeptex_arms_report.py` | dataprep container: several arms (`--arm label=dir`) against LR / bilinear / MatUp / GT on every map; montages, overviews, sheets, `summary.md`, `per_material.md`, `metrics_long.csv`, `report.json` |
| `deeptex_drift.py` | deeptex container: per checkpoint, mean(SR) - mean(HR) per channel on train pairs and on the eval materials (the colour cast); `--sheet-material` draws that material's six outputs per checkpoint |

## How it works

```
/data/MatSynth/data/train-*.parquet
  -> matsynth_make_pairs.py      runs/deeptex/datasets/matsynth/<map>/{HR,LRx4}/<material>__<map>.png
                                 + _meta/<material>.json + pairs_manifest.json
  -> recipes.py --recipe R       .../matsynth/matsynth_<R>_lrx4.tfrecord (+ .meta.json)
  -> upstream_runner/texsr_verify_tfrecord.py --align-sample 16
  -> paper/run.ps1 -Stage paper -PaperConfig <cfg> -PaperRoot runs/deeptex/<arm>
                                 runs/deeptex/<arm>/checkpoints/esrgan_40k/
  -> eval_arms.py                runs/deeptex/eval/<tag>/<m>/{baseColor,normal,roughness,metallic,height,render}.png
                                 + sr_manifest.json (incl. channel_spread of the 1-channel maps)
  -> deeptex_arms_report.py      runs/deeptex/report_<tag>/
```

The pairs use the same degradation and encoding as the evaluation set: `degrade()`
calls `variants/util/matsynth_extract.py`'s functions in the order its `--mode resize`
uses (area-decimate the source to 512^2 for HR and separately to 128^2 for LR), then
`variants/util/map_encode.to_rgb8`, which `infer/deeptex_compare.load_as_rgb01` also
uses. `--check-eval` proves it from `/data/MatSynth/data/test-00000-of-00008.parquet`.

The recipes (`recipes.RECIPES`): times each map folder is packed into the ONE record,
with one seeded shuffle over all of it, because upstream reads with `shuffle=False`
and the record order is the curriculum. Copies after the first get `~r2`..`~rk`
appended to `img_name`.

| recipe | baseColor | normal | roughness | metallic | height | config | arm root |
|---|---|---|---|---|---|---|---|
| `all` (the paper's mix) | 1 | 1 | 1 | 1 | 1 | `esrgan_matsynth.yaml` | `runs/deeptex/matsynth_all` |
| `chratio` (ablation, channel count) | 3 | 3 | 1 | 1 | 1 | `esrgan_matsynth_chratio.yaml` | `runs/deeptex/matsynth_chratio` |
| `basecolor` (control) | 1 | - | - | - | - | `esrgan_matsynth_basecolor.yaml` | `runs/deeptex/matsynth_basecolor` |

The three configs differ from `paper/esrgan_paper.yaml` only in their header comment
and `train_dataset.path` (`/workspace/runs/deeptex/datasets/matsynth/matsynth_<recipe>_lrx4.tfrecord`);
`sub_name` stays `esrgan_40k` and the arms are kept apart by working directory. Each
header carries its own `diff --strip-trailing-cr <(sed '1,Nd' ...)` check against
upstream, and it shows the single line. `run.ps1` reads `sub_name` with the line regex
`^sub_name:\s*'(.+)'`.

Stages of `run.ps1` (`-Stage` is mandatory; logs in `runs/_logs/<stamp>/`):

| stage | runs |
|---|---|
| `preflight` | `Invoke-DeeptexPreflight` (`variants/util/lib.ps1`); runs before every stage unless `-SkipPreflight` |
| `prep-matsynth` | `matsynth_make_pairs.py --check-eval` (a hard gate) -> `matsynth_make_pairs.py --out runs/deeptex/datasets/matsynth --jobs 4` (resumes from `_meta/`) -> `recipes.py --recipe <Recipe> --root ... --overwrite` -> `texsr_verify_tfrecord.py --align-sample 16` |
| `eval-arms` | `eval_arms.py` with `<PaperConfig>` and `<PaperRoot>/checkpoints/<sub_name>` -> `runs/deeptex/eval/<tag>`; `eval_arms.py` with the textures_db checkpoint (`runs/deeptex/paper/checkpoints/esrgan_40k`) -> `runs/deeptex/eval/paper_allmaps`; `deeptex_arms_report.py --arm textures_db=... [--arm <ExtraArms>...] --arm <tag>=...` -> `runs/deeptex/report_<tag>/summary.md` |

`<tag>` is `-EvalTag` if given, else the leaf of `-PaperRoot`. `eval-arms` writes only
under `runs/deeptex/eval/` and `runs/deeptex/report_<tag>/`, never into
`runs/matsynth/<m>/compare/` or `runs/montages/`.

What it reads from other folders:

- **MatUp baseline**: `eval_arms.py`'s `render` job super-resolves MatUp's
  `render_lr.png` from the newest `runs/matsynth/<m>/matup-output-<m>/<stamp>/`.
  `deeptex_arms_report.py` builds its lr / bilinear / MatUp / GT columns from
  `runs/matsynth/<m>/` with `matup/baseline/postprocess.py` (same crop, border and
  DirectX -> OpenGL flip as `runs/montages/`) and draws with `matup/baseline/montage.py`
  / `util/report/overview.py`.
- **Other arms**: `deeptex_arms_report.py --arm` takes any `eval_arms.py --out`; the
  render-loss arms are `runs/deeptex/eval/render_r*_<k>k`.
- **Pairs**: `runs/deeptex/datasets/matsynth/<map>/{HR,LRx4}` (and `_meta/`) also feed
  the grouped record of `render_loss/build/`.
- `eval_arms.py` re-exports `load_as_rgb01` and `newest_run`; `deeptex_drift.py`
  imports it as `deeptex_compare` and uses `JOBS`, `save_sr` and those two.

## Run

From the repo root (`powershell -NoProfile -ExecutionPolicy Bypass -File variants\deeptex\matsynth\run.ps1 ...`):

```
# 1. pairs + record for one recipe
variants\deeptex\matsynth\run.ps1 -Stage prep-matsynth -Recipe chratio

# 2. train (in paper/run.ps1)
variants\deeptex\paper\run.ps1 -Stage paper -PaperConfig variants/deeptex/matsynth/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all

# 3. all-maps evaluation + report
variants\deeptex\matsynth\run.ps1 -Stage eval-arms -PaperConfig variants/deeptex/matsynth/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all
```

`-ExtraArms a,b` adds already-evaluated `runs/deeptex/eval/<a>` folders as report
columns (checked before any SR pass). Without `-PaperConfig` / `-PaperRoot`,
`eval-arms` evaluates the textures_db checkpoint under the tag `paper`.

The drift check has no stage; run it directly:

```
docker compose run --rm deeptex python /workspace/variants/deeptex/matsynth/deeptex_drift.py \
    --cfg      /workspace/variants/deeptex/matsynth/esrgan_matsynth.yaml \
    --ckpt-dir /workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k \
    --pairs    /workspace/runs/deeptex/datasets/matsynth/baseColor \
    --out      /workspace/runs/deeptex/drift/matsynth_all
```

Point `--ckpt-dir` at `checkpoints_all/esrgan_40k` (from `paper/run.ps1
-KeepAllCheckpoints`) to get every checkpoint. It writes `drift.json`, `drift.md`
and, with `--sheet-material`, `<material>_<ckpt>_<when>.png` sheets.

## Known behaviour and defects

- `esrgan_matsynth_chratio.yaml`'s header says the record is built with
  `texsr_build_tfrecord.py --repeat` and refers to `run_deeptex.ps1 -PaperRoot`;
  neither exists - the record is `recipes.py --recipe chratio`, training is
  `paper/run.ps1 -PaperRoot`. The basecolor and all headers name `run_deeptex.ps1` too.
- `deeptex_drift.py` hard-codes scale 4, whole-image inference
  (`super_resolve(..., 4, 0, 16)`) and the `"mean"` collapse for its sheet, whatever
  the config says.
- The verify step of `prep-matsynth` reports a nonzero `max |lr - imresize_np(hr)|` as
  a warning by design: LR here is area-decimated from the 4K source, not
  `imresize_np(HR)`.
- A metallic GT is a constant 0 on 9 of the 12 materials; a column that reproduces it
  has infinite PSNR. `deeptex_arms_report.py` averages finite values and reports the
  count of exact reproductions beside them.
- GPU inference is not bit-reproducible (DEEPTEX-DEVIATIONS 0.9): a rerun moves
  outputs by up to 1 LSB, which can move a PSNR in the second decimal.
- Height is 16-bit in MatSynth and becomes 8-bit in the pairs (upstream decodes PNGs
  as uint8); grey maps are replicated to three channels; constant maps are kept.
- Some docstrings cite `scripts/...` / `configs/...` paths and line numbers that no
  longer exist in the repo.
