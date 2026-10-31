# DeepTextures eval tree — equivalence tests (L0 / L2 / L3)

What is tested: everything of DeepTextures except training - data prep
(`upstream_runner/`, `paper/`, `matsynth/`, `render_loss/build/`), inference
(`infer/`, `matsynth/eval_arms.py`), evaluation and reports (`compare/`,
`matsynth/deeptex_arms_report.py`, `matsynth/deeptex_drift.py`, `tiling/`,
`render_loss/eval/`) and the MatUp modules they import (`matup/baseline/matup_{render,height}.py`).

## Run

From the main checkout (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File variants\tests\run_deeptex_eval.ps1
# a worktree (its runs\ and external\ are empty): mount the main checkout's
powershell -ExecutionPolicy Bypass -File variants\tests\run_deeptex_eval.ps1 -Main C:\Users\admin\Desktop\Vsc_Workspace\TexSR-comp1
# options: -Layers L0,L2 | -Quick (L2 on 300-material subsets) | -NoGpu | -Data D:\experimentdata | -Stamp <name>
```

Output: `runs\variants_verify\deeptex_eval\<stamp>\report.md` (per layer PASS/FAIL, the
failing diffs, and the committed files that could not be matched), with the per-layer
JSON (`l2\l2_report.json`, `l3\l3_infer.json`, `l3\l3_reports.json`), `l0.md`, and
every step's log under `l2\logs\`, `l3\logs\`. A layer can be rerun alone with the same
`-Stamp`; its rows replace the earlier ones.

Single layers, by hand (host Python 3.8+, docker):

```
python variants/tests/l0_static.py                                  # seconds
python variants/tests/l2_data.py --stamp S --main <main> --data D:/experimentdata [--only tdb,ms_rec,grouped,ms_pairs]
python variants/tests/l3_infer.py --stamp S --main <main> [--no-gpu]
python variants/tests/l3_reports.py --stamp S --main <main> [--only montages,base,arms,render,perceptual,drift,tiling]
python variants/tests/l2_harness.py report runs/variants_verify/deeptex_eval/S
```

Every step is a `docker run` of the project images with the checkout mounted at
`/workspace` **read-only**, `runs/` and `external/` read-only on top, and only
`runs/variants_verify/` writable (`l2_harness.py`). GPU steps (`--gpus all`) first wait
until no other `texsr*` container is running.

## What each layer proves

| layer | file | how |
|---|---|---|
| L0 | `l0_static.py` + `moves/deeptex_eval.json` | moved files: AST-identical; split functions: every old line verbatim, rewritten (to a line that exists) or dropped with a reason |
| L2 data | `l2_data.py`, `l2_records.py` | old and new builders into the SAME path (the old output moved aside): records equal record by record and in order (`order_sha256`), not by file sha (protobuf map order); meta, verify reports, manifests and pair PNGs equal |
| L3 inference | `l3_infer.py`, `l3_pngdiff.py` | old and new `deeptex_compare` / `eval_arms` on the CPU, same checkpoint, same `--out`: byte-identical. GPU: new vs the committed eval folders, <= 1 LSB (reference) |
| L3 reports | `l3_reports.py` | the new report layer on the frozen `runs/`: JSON numerically, md/csv as text, against the committed files; the old code on the same inputs too, so a committed file that predates its inputs shows up as "stale" (old == new != committed) instead of as a refactoring error |

### L2 in detail

| section | old | new | input |
|---|---|---|---|
| `tdb` | `texsr_normalize` -> `texsr_make_pairs` -> `texsr_build_tfrecord` -> `texsr_verify_tfrecord` | the same four under `variants/` | `external/DeepTextures/data/textures_db`, all 299 sources (2965 + 264 tiles) |
| `ms_rec` | `texsr_build_tfrecord --hr/--lr x5 [--repeat]` as run_deeptex.ps1 passed them, + verify | `matsynth/recipes.py --recipe all/chratio/basecolor`, + verify | `runs/deeptex/datasets/matsynth`, all 2904 materials (`-Quick`: first 300 per map) |
| `grouped` | `texsr_build_grouped_tfrecord`, `texsr_verify_grouped` | `render_loss/build/` | the same pair root, all 2904 (`-Quick`: `--limit 300`) |
| `ms_pairs` | `matsynth_make_pairs --check-eval`; `--limit-shards 1` | the same | `/data/MatSynth/data` (`-Data`) |

`ms_pairs` covers ONE train shard, not the 431 the committed pairs came from (hours);
the PNG writer is the same code for every shard. `pairs_manifest.json` of
`texsr_make_pairs` is compared as JSON, because its key order is the multiprocessing
completion order (two runs of the old code differ in bytes).

### L3 reports in detail

| section | regenerated | compared with |
|---|---|---|
| `montages` | `paper/deeptex_postprocess --deeptex-root runs/deeptex/eval/paper` + `paper/deeptex_montage --collect` on a copy of `runs/matsynth` | `runs/montages/<material>.json` (12), `runs/montages/summary.md`; old vs new: every file of `collect/` and `<mat>/compare/` |
| `base` | the same without the DeepTextures column (`compare/*`) | old vs new, byte for byte |
| `arms` | `matsynth/deeptex_arms_report` with the arms recorded in each committed report.json | `runs/montages/{deeptex_matsynth,deeptex_data_ratio,deeptex_render_loss}/{report.json,metrics_long.csv,summary.md,per_material.md}` |
| `render` | `render_loss/eval/svbrdf_render_eval` (GPU, as committed: `constants.device = cuda:0`) | `runs/montages/deeptex_render_loss/render_summary.md`, `runs/deeptex/render_eval/main/{report.json,per_material.md,metrics_long.csv}` |
| `perceptual` | `render_loss/eval/svbrdf_perceptual_eval` (GPU) | `runs/montages/deeptex_render_loss/{perceptual_report.json,perceptual_summary.md}` |
| `drift` | `matsynth/deeptex_drift`: CPU old vs new (matsynth_all, 3 checkpoints, `--n 6`); GPU new for `matsynth_all`, `render_r1_baseColor` | CPU: byte-identical; GPU: committed `drift.md` / `drift.json`, informational (GPU 1-LSB) |
| `tiling` | `tiling/deeptex_tiling_report` | `runs/tiling/tiling_report.json` |

Not regenerated: the montage folders' `README.md` (written by run_deeptex.ps1, not by
the report layer), `runs/tiling/summary.md` (hand-written), and `dt_only/` of
`deeptex_render_loss` (`deeptex_dt_montage.py`; its README prints the script path,
which changed by design - the file is L0-identical otherwise).

## Time

Measured on this machine (20 logical CPUs, RTX 3080 Ti 12 GB, Docker Desktop on Windows, bind mounts):

| step | time | of which |
|---|---|---|
| L0 | 5 s | |
| L2 | 50 min | `tdb` ~10 min (each chain ~4.7 min: normalize 1, pairs 3, record+verify 1); `ms_rec` ~30 min (all, chratio, basecolor: build + verify, old and new, + record compare; chratio alone ~14 min, 26 136 records); `grouped` ~8 min; `ms_pairs` ~1 min |
| L3 inference | 6 min | CPU paper 2 x 50 s, CPU all-maps 2 x ~3 min, GPU reference ~1 min |
| L3 reports | 26 min | `arms` ~8 min, `montages` + `base` ~6 min, `render` 2 x 2 min (GPU), `perceptual` 2 x 3 min (GPU), `drift` CPU 2 x 2 min + GPU 2 x 2.5 min, `tiling` 40 s |
| **total** | **81 min** | `-Quick` cuts L2 to ~15 min |

Peak extra disk under `runs\variants_verify\`: ~12 GB while the chratio records exist
(each record is deleted once compared). The 20261031-final run left 4.4 GB (the
textures_db pair folders, the `arms` montages, the `montages`/`base` collects).

Last full run: `runs\variants_verify\deeptex_eval\20261031-final\report.md` - L0 123/0 FAIL,
L2 23/0, L3 inference 5/0, L3 reports 48/0 (3 of them "stale": `deeptex_matsynth`'s
committed report.json / metrics_long.csv / per_material.md predate the 2026-10-02 rerun of
`runs/deeptex/eval/paper_allmaps`; all 579 differing values are in its `dt_textures_db`
column, old code == new code).
