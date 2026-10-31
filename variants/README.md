# variants/

Every experiment run on top of the unmodified submodules (DeepTextures `e4ee562`,
matup_egsr `c74ec99`), one folder per experiment. The old code in `scripts/` and
`experiments/` stays as it is; this tree reimplements it and the tests in
`variants/tests/` prove the two behave the same.

> Work in progress on branch `refactor/variants`. The experiment → folder →
> change table is filled in as each part lands (plan step 7).

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

## Layout (owner while the work is split across sessions)

| folder | contents |
|---|---|
| `common/` | `paths.py`, moved helpers (`map_encode`, `matsynth_extract`, `matup_lights`, `texsr_metrics`), `lib.ps1`, `preflight.ps1` |
| `deeptex/paper_loop/`, `hooks_common/`, `fixed/`, `render_loss/` (trainer, gates) | DeepTextures trainer = the paper loop plus per-experiment hooks |
| `deeptex/upstream_runner/`, `infer/`, `paper/`, `matsynth/`, `tiling/`, `render_loss/eval/`, `render_loss/build/`, `compare/` | data prep, inference, evaluation, reports |
| `matup/baseline/` | MatUp baseline |
| `experiments/` | experiments 1-5 |
| `tests/` | equivalence tests, `run_all.ps1` |
