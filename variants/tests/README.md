# variants/tests/

What is left here is the one check that does not need the old code:

```
python variants/tests/l0_skeleton.py           # loop.py = upstream train_esrgan.py + hook calls only
python variants/tests/l0_skeleton.py --write   # regenerate paper_loop/upstream.diff, psnr_upstream.diff
```

It compares `variants/deeptex/paper_loop/loop.py` and `psnr_loop.py` against the
submodules' `train_esrgan.py` / `train_psnr.py`, checks that every upstream line
a hook call replaced is kept in `paper_loop/hooks.py`, and that the committed
`.diff` files are current. Run it after any change to `paper_loop/`.

## The equivalence tests (removed)

The layers below proved the new tree behaves like the old `scripts/`,
`experiments/` and `configs/`. Once they all passed, the old code and these
tests were deleted together. Both are at **tag `variants-verified`**.

| layer | what it proved |
|---|---|
| L0 static | moved code AST-identical; every line of a split function verbatim, renamed, rewritten or dropped with a reason (`moves/*.json`) |
| L1 trainer | old `deeptex_train.py` vs `main.py --experiment ...`, and `train_esrgan.py` vs `--experiment paper`: same exit code, stdout, checkpoint (CPU, deterministic harness `_det_run.py`) |
| L1 GPU | full-size r1, one step: `[init]` identical, step numbers within 10x the old-vs-old spread |
| L2 / L3 | data records, inference, reports identical to the old code and to the committed files |
| L4 | render gates: gate JSONs, `parity_tf.npz`, calibration (w* = 10) |
| L5 | every orchestrator stage with docker intercepted: same docker argv, transcript, created files |
| L6 | experiments 1-5: configs, state_dict keys, losses and grads, GPU optimise/smoke/compat |

Last full pass: `run_all.ps1` stamp `20261101-004941` on `962ffbe`, all eight
layers PASS in 128 min. Reports (kept):

- `runs/variants_verify/20261101-004941/report.md` and the per-layer `L*.log`
- `runs/variants_verify/deeptex_eval/20261101-004941-L2/`, `-L3/` (L2, L3)
- `runs/variants_verify/l1/`, `l4/`, `deeptex_l5/`, `matup/`, `experiments/20261101_024032/`

To run them again, check the tag out in the main checkout (they need its
`runs/` data and write under `runs/variants_verify/`), then come back:

```
git switch --detach variants-verified
powershell -NoProfile -ExecutionPolicy Bypass -File variants\tests\run_all.ps1
git switch -
```

## GPU runs are not bit-reproducible: measured spreads

Kept because it holds for any GPU comparison on this machine. The old code run
twice on the RTX 3080 Ti (2026-10-31 / 2026-11-01):

| what | old vs old | new vs old |
|---|---|---|
| r1, one step, max relative difference over `[terms]`/`[render]` | 1.85e-3 / 9.4e-4 | 7.0e-4 / 2.8e-4 |
| render calibration, relative difference over ~1,765 numbers: median / p90 | 0.00127 / 0.0166 | 0.00123 / 0.0166 |
| render calibration, unrounded w* | 9.553 / 9.565 / 9.566 / 9.561 | 9.565 / 9.554 |

The calibration's largest relative difference (0.9-1.4) is a per-batch cosine
near zero that changes sign between runs of the same code. w* rounds to 10 on
every run.
