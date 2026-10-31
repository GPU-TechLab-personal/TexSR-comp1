# variants/tests/ - equivalence of the new tree with the old code

Everything runs on the desktop (GPU, the `texsr/*` images, `runs/deeptex`,
`runs/matsynth`). One command runs every layer:

```
powershell -NoProfile -ExecutionPolicy Bypass -File variants\tests\run_all.ps1
```

It writes `runs/variants_verify/<stamp>/report.md` (per-layer PASS/FAIL, the
failing diffs). Every test writes only under `runs/variants_verify/` (and the
experiments under `/expdata/variants_verify/`); nothing in the existing `runs/`
is modified. This is the only folder in `variants/` that imports the old code.

| layer | script | needs | what it proves |
|---|---|---|---|
| L0 static | `l0_static.py` + `moves/*.json` | host Python | moved code is AST-identical (imports/docstrings aside); split functions: every old line is verbatim, a listed rename rule, a listed rewrite, or a listed drop with its reason |
| L0 skeleton | `l0_skeleton.py` | host Python | `paper_loop/loop.py` = `train_esrgan.py` + hook calls only (and `psnr_loop.py` = donor `train_psnr.py`); every replaced upstream line is kept in `paper_loop/hooks.py`; the committed `.diff`s are current |
| L1 trainer | `l1_fixtures.py`, `l1_trainer.py` | deeptex image, CPU | old `deeptex_train.py` vs `main.py --experiment ...`, and `train_esrgan.py` vs `--experiment paper`: same exit code, same stdout line for line, bit-identical checkpoint |
| L1 GPU | `l1_trainer.py --gpu` | GPU | full-size r1, one step, old twice + new: `[init]` identical on all three, `[terms]`/`[render]` new-vs-old within max(1e-6, 10x old-vs-old) |
| L2 / L3 | `run_deeptex_eval.ps1` (`l2_*.py`, `l3_*.py`; see `README_deeptex_eval.md`) | images, `runs/` | data records, reports, inference |
| L4 gates | `l4_gates.py` | GPU, deeptex + matup images | gate JSONs exact; `parity_tf.npz` array by array; calibration: old twice, median/p90/max relative difference new-vs-old within 10x old-vs-old, w* = 10 everywhere |
| L5 orchestrators | `l5_deeptex.ps1`, `l5_matup.ps1` | PowerShell | every `-Stage` of the old ps1 (48 deeptex scenarios over all 15 stages, 13 matup) vs the arm `run.ps1`, docker intercepted: same docker argv, transcript and created files after the mapping |
| L6 experiments | `l6_*.py` (see `variants/experiments/README.md`) | exp image | exp1-5 arms, configs, state_dict keys, gates |

## L1: how determinism is obtained (and where it is not)

`_det_run.py` runs each trainer unmodified with `runpy`, after:

* `tf.config.experimental.enable_op_determinism()`;
* every `tf.data.Dataset.map` made sequential - upstream's loader
  (`modules/dataset.py:99-101`) draws its crops with stateful `tf.random.uniform`
  inside a parallel map, so which tile gets which crop depended on thread timing
  (upstream against itself diverged at step 4 without this);
* intra- and inter-op threads set to 1 - with op determinism alone, upstream
  against itself still differed by 1e-8..1e-6 after five CPU steps;
* the same seed (`--seed 7`; `--paper-seed 3` for `train_esrgan.py`, which never
  seeds), `TF_ENABLE_ONEDNN_OPTS=0`, `PYTHONHASHSEED=0`.

`L1_selfcheck_upstream` (upstream against itself) runs first; if it fails,
nothing after it means anything.

Measured exception: **from scratch with the render term on** (`render_r1`,
`render_r1_resume_3_5`) the OLD trainer is not reproducible against itself even
under all of the above - the step on the third grouped material varies run to
run (three old runs: `[terms] render` 5.1079 / 5.1079 / 5.108, `[render]` step-4
l21 0.848005 / 0.847927 / 0.848328). Those two cases run the old trainer twice
and pass when new-vs-old is within 10x old-vs-old, for every logged number and
for the checkpoint's max |d|; the report prints both spreads. The same render
term over two steps, and with one cached batch (`render_overfit_from_pretrain`,
w = 9.54), IS bit-identical old vs new.

Masked in stdout: the output root (`<R>`), `steps_per_sec`, `eta_h`, the disk-free
GB and the progress bar's step/sec. Nothing else.

## Time (desktop, RTX 3080 Ti, measured 2026-10-31)

| layer | time |
|---|---|
| L0 (both) | < 5 s |
| L1 CPU, 16 cases | ~25 min (render cases dominate; single-threaded on purpose) |
| L1 GPU | ~3 min (three one-step runs) |
| L2 + L3 | 81 min (`run_deeptex_eval.ps1`, see `README_deeptex_eval.md`) |
| L4 | ~25 min (parity twice, calibration three times at ~4 min) |
| L5 (deeptex + matup) | ~1 min |
| L6 | ~21 min (GPU ~17) |

## GPU runs are not bit-reproducible: measured spreads

The GPU layers compare against the old code run twice, not against one old run.
Measured 2026-10-31 on the RTX 3080 Ti:

| what | old vs old | new vs old |
|---|---|---|
| L1 GPU, r1 step 1, max relative difference over `[terms]`/`[render]` | 1.85e-3 | 7.0e-4 |
| L4 calibration, relative difference over 1,765 numbers: median / p90 / max | 0.00127 / 0.0166 / 1.31 | 0.00123 / 0.0166 / 1.31 |
| L4 calibration, unrounded w* | 9.553 / 9.565 (committed 9.541) | 9.565 |

The max of 1.31 is a per-batch cosine near zero (`states[0]/per_batch/cos_offrm_feature[12]`:
0.0344 in the first old run, -0.1099 in the second old run and in the new one).
w* rounds to 10 on every run.
`parity_tf.npz` changes sha256 on every export only because its manifest records
the time; its 63 arrays are identical.
