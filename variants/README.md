# variants/

Every experiment run on top of the unmodified submodules (DeepTextures `e4ee562`,
matup_egsr `c74ec99`), one folder per experiment. The old code in `scripts/` and
`experiments/` stays as it is; this tree reimplements it and the tests in
`variants/tests/` prove the two behave the same.

## Experiment → folder → what changed

| experiment | folder | run it with | what changed against the old code |
|---|---|---|---|
| DeepTextures paper arm (textures_db, upstream `train_esrgan.py` as released) | `deeptex/paper/`, `deeptex/upstream_runner/`, `deeptex/paper_loop/` | `deeptex/paper/run.ps1 -Stage prep\|smoke\|paper\|eval\|all` | `paper_loop/loop.py` is `train_esrgan.py` plus hook calls only (`upstream.diff`); the paper stage still execs the submodule's own trainer through `run_upstream_train.py`. The fifth comparison column moved out of `matup_postprocess` into `paper/deeptex_postprocess.py` / `deeptex_montage.py` |
| DeepTextures on MatSynth (all / chratio / basecolor) | `deeptex/matsynth/` | `deeptex/matsynth/run.ps1 -Stage prep-matsynth\|eval-arms`; training: `paper/run.ps1 -Stage paper -PaperConfig variants/deeptex/matsynth/<cfg>.yaml -PaperRoot runs/deeptex/<arm>` | the recipe table and `--repeat` became `recipes.py --recipe`; `deeptex_compare --maps all --scalar-collapse` became `eval_arms.py` |
| DeepTextures fixed arm (defects #2-#5, #8 corrected) | `deeptex/fixed/` | `deeptex/fixed/run.ps1 -Stage pretrain\|train` | `deeptex_train.py`'s fixed branches became `fixed/hooks.py` on the paper loop (`paper_loop/main.py --experiment fixed`, same flags) |
| Render loss (R0a / R0b / R1) | `deeptex/render_loss/` | `deeptex/render_loss/run.ps1 -Stage prep-grouped\|gate-render\|render-train\|eval-render\|eval-perceptual` | the grouped/render branches became `render_loss/hooks.py` + `terms.py` (`--experiment render`); gates, record builder and render eval moved |
| Tiling study | `deeptex/tiling/` | `python variants/deeptex/tiling/deeptex_tiling_report.py` | moved |
| MatUp baseline | `matup/baseline/` | `matup/baseline/run.ps1` | moved; the shared comparison/montage code is `compare/` |
| Experiments 1-5 | `experiments/exp*/`, `experiments/matup_driver/`, `experiments/gates/` | see `experiments/README.md` §5 | exp2 split into `core/` + one module per arm, exp3 arms on the exp2 trainer, `matup_driver` out of `common/` |

Each folder's README lists its files against the old ones, the odd behaviour it
keeps, and the defects found but not fixed. The orchestrators: every stage of
`scripts/deeptex/run_deeptex.ps1` now lives in exactly one arm `run.ps1`, with
the same stage name and parameters; the helpers they share are
`common/lib.ps1` (L5 compares their docker argv stage by stage).

## Rules every file here follows

1. **Self-contained.** Nothing under `variants/` imports `scripts/` or
   `experiments/`; the only outside code is `external/*`. The one exception is
   `variants/tests/`, whose equivalence tests run the old code next to the new.
2. **Package imports, one bootstrap.** Entry scripts put the repo root on
   `sys.path` and import by package:

   ```python
   import os, sys
   sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * N)))
   from variants.common import paths      # N = depth of this file below the repo root
   paths.use_deeptex()                    # or use_matup() / use_swinir()
   ```

   Libraries import siblings as `from variants.<...> import x`, never through a
   `sys.path` edit. Submodule paths, the DEEPTEX_ROOT override and the pin check
   live only in `common/paths.py`.
3. **Moved vs decomposed.** A file without experiment branches is copied and only
   its import lines / path constants change; L0 proves the AST is identical. Only
   files with experiment branches are actually split, and L0 then checks that
   every line of the old function is accounted for (verbatim, rewritten, or
   dropped with a reason). Each area records its entries in its own
   `variants/tests/moves/<area>.json`.
4. **External contracts stay byte-identical.** Log prefixes (`[metrics]`,
   `[terms]`, `[render]`, `[init]`, `[*] resumed from`, `[fatal] non-finite loss`),
   exit codes 2-9, checkpoint object names, top-level config keys (the ps1 files
   read them with line regexes), exp2 `state_dict` keys. Existing checkpoints and
   eval folders must load unchanged.
5. **Odd behaviour is kept and named, not fixed.** It is made explicit in the arm
   module and written down in that arm's README.
6. **No folder named `data/`.** `.gitignore:6` ignores it (that is why
   `experiments/data/*.py` was never committed).
7. **Tests write only under `runs/variants_verify/`.** Nothing in the existing
   `runs/` is modified.
8. **One GPU job at a time.** Before any GPU step, check that no other `texsr*`
   container is running (`docker ps`); docs/RUNBOOK.md's concurrency warning applies.

## Layout

| folder | contents |
|---|---|
| `common/` | `paths.py`, moved helpers (`map_encode`, `matsynth_extract`, `matup_lights`, `texsr_metrics`), `lib.ps1`, `preflight.ps1` |
| `deeptex/paper_loop/`, `hooks_common/`, `fixed/`, `render_loss/` (trainer, gates) | DeepTextures trainer = the paper loop plus per-experiment hooks |
| `deeptex/upstream_runner/`, `infer/`, `paper/`, `matsynth/`, `tiling/`, `render_loss/eval/`, `render_loss/build/`, `compare/` | data prep, inference, evaluation, reports |
| `matup/baseline/` | MatUp baseline |
| `experiments/` | experiments 1-5 |
| `tests/` | equivalence tests, `run_all.ps1` |
