# variants/

Every experiment run on top of the unmodified submodules (DeepTextures `e4ee562`,
matup_egsr `c74ec99`), one folder per experiment. It reimplements the old
`scripts/`, `experiments/` and `configs/`, which were deleted once the
equivalence tests (L0-L6) passed against them.

**Tag `variants-verified`** is the last commit with the old code and the tests
next to this tree (full pass `20261101-004941`, all eight layers PASS). The
READMEs here still name old files, test layers (L0-L6), `l*_*.py` scripts and
`variants/tests/moves/*.json` as the record of what was moved and proven: read
them with `git show variants-verified:<path>`. Line numbers in old paths refer
to that tag.

## Experiment → folder → what changed

| experiment | folder | run it with | what changed against the old code |
|---|---|---|---|
| DeepTextures paper arm (textures_db, upstream `train_esrgan.py` as released) | `deeptex/paper/`, `deeptex/upstream_runner/`, `deeptex/paper_loop/` | `deeptex/paper/run.ps1 -Stage prep\|smoke\|paper\|eval\|all` | `paper_loop/loop.py` is `train_esrgan.py` plus hook calls only (`upstream.diff`); the paper stage still execs the submodule's own trainer through `run_upstream_train.py`. The fifth comparison column moved out of `matup_postprocess` into `paper/deeptex_postprocess.py` / `deeptex_montage.py` |
| DeepTextures on MatSynth (all / chratio / basecolor) | `deeptex/matsynth/` | `deeptex/matsynth/run.ps1 -Stage prep-matsynth\|eval-arms`; training: `paper/run.ps1 -Stage paper -PaperConfig variants/deeptex/matsynth/<cfg>.yaml -PaperRoot runs/deeptex/<arm>` | the recipe table and `--repeat` became `recipes.py --recipe`; `deeptex_compare --maps all --scalar-collapse` became `eval_arms.py` |
| DeepTextures fixed arm (defects #2-#5, #8 corrected) | `deeptex/fixed/` | `deeptex/fixed/run.ps1 -Stage pretrain\|train` | `deeptex_train.py`'s fixed branches became `fixed/hooks.py` on the paper loop (`paper_loop/main.py --experiment fixed`, same flags) |
| Render loss (R0a / R0b / R1) | `deeptex/render_loss/` | `deeptex/render_loss/run.ps1 -Stage prep-grouped\|gate-render\|render-train\|eval-render\|eval-perceptual` | the grouped/render branches became `render_loss/hooks.py` + `terms.py` (`--experiment render`); gates, record builder and render eval moved |
| Tiling study | `deeptex/tiling/` | `python variants/deeptex/tiling/deeptex_tiling_report.py` | moved |
| MatUp baseline | `matup/baseline/` | `matup/baseline/run.ps1` | moved; the shared comparison/montage code is `compare/` |
| Experiments 1-5 | `experiments/exp*/`, `experiments/matup_driver/`, `experiments/gates/` | `experiments/EXPERIMENTS.md` (commands), `experiments/README.md` §5 | exp2 split into `core/` + one module per arm, exp3 arms on the exp2 trainer, `matup_driver` out of `common/` |

Each folder's README lists its files against the old ones, the odd behaviour it
keeps, and the defects found but not fixed. The orchestrators: every stage of
`scripts/deeptex/run_deeptex.ps1` now lives in exactly one arm `run.ps1`, with
the same stage name and parameters; the helpers they share are
`util/lib.ps1` (L5 compares their docker argv stage by stage).

## Rules every file here follows

1. **Self-contained.** The only outside code is `external/*` (the old
   `scripts/` and `experiments/` no longer exist).
2. **Package imports, one bootstrap.** Entry scripts put the repo root on
   `sys.path` and import by package:

   ```python
   import os, sys
   sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * N)))
   from variants.util import paths      # N = depth of this file below the repo root
   paths.use_deeptex()                    # or use_matup() / use_swinir()
   ```

   Libraries import siblings as `from variants.<...> import x`, never through a
   `sys.path` edit. Submodule paths, the DEEPTEX_ROOT override and the pin check
   live only in `util/paths.py`.
3. **Moved vs decomposed.** A file without experiment branches is copied and only
   its import lines / path constants change; L0 proves the AST is identical. Only
   files with experiment branches are actually split, and L0 then checks that
   every line of the old function is accounted for (verbatim, rewritten, or
   dropped with a reason). Each area recorded its entries in its own
   `variants/tests/moves/<area>.json` (tag `variants-verified`).
4. **External contracts stay byte-identical.** Log prefixes (`[metrics]`,
   `[terms]`, `[render]`, `[init]`, `[*] resumed from`, `[fatal] non-finite loss`),
   exit codes 2-9, checkpoint object names, top-level config keys (the ps1 files
   read them with line regexes), exp2 `state_dict` keys. Existing checkpoints and
   eval folders must load unchanged.
5. **Odd behaviour is kept and named, not fixed.** It is made explicit in the arm
   module and written down in that arm's README.
6. **No folder named `data/`.** `.gitignore:6` ignores it (that is why
   `experiments/data/*.py` was never committed).
7. **Checks write only under `runs/variants_verify/`.** Nothing in the existing
   `runs/` is modified by a check.
8. **One GPU job at a time.** Before any GPU step, check that no other `texsr*`
   container is running (`docker ps`); docs/RUNBOOK.md's concurrency warning applies.

## Adding an experiment

A new experiment must leave the existing arms unchanged, and gets its own
smoke run.

### DeepTextures: a new training variant (e.g. one more loss term)

`paper_loop/main.py --experiment <name>` looks the name up in `EXPERIMENTS`
(`main.py:26`); that module's `build(argv)` returns `(hooks, "loop" | "psnr_loop")`,
and `loop.py` calls the hooks listed at the top of `paper_loop/hooks.py`. The
defaults are the paper; an experiment overrides only what it changes
(`fixed/hooks.py` overrides two). Extra G terms come from `extra_loss`: the dict
it returns goes into `losses_G` (`loop.py:84`) and is summed with reg, pixel,
feature and gan.

1. `variants/deeptex/<name>/`: `hooks.py`, a config yaml, `run.ps1`, `README.md`.
2. `hooks.py` - subclass `hooks_common.trainer.TrainerHooks` (checkpoints,
   `[metrics]`, exit codes come with it):

   ```python
   import tensorflow as tf
   from variants.deeptex.hooks_common.trainer import TrainerHooks, add_common_args, parse

   def add_args(ap):
       add_common_args(ap)                      # --cfg_path, --max-steps, --seed, ...
       ap.add_argument("--stage", default="esrgan", choices=["esrgan"])
       ap.add_argument("--w-tv", type=float, default=0.0)

   class MyHooks(TrainerHooks):
       def extra_loss(self, sr, hr, extras):
           if self.args.w_tv > 0:              # a Python float: a constant in the trace
               return {"tv": self.args.w_tv * tf.reduce_mean(tf.image.total_variation(sr))}
           return {}

       def extra_report(self, steps, losses_G, outs):
           # [metrics] prints only the sum; log.terms_line knows the render arm's keys only.
           if "tv" in losses_G and (steps == 1 or steps % self.args.log_every == 0):
               print("[tv] step=%d tv=%.5g" % (steps, float(losses_G["tv"].numpy())), flush=True)

   def build(argv):
       return MyHooks(parse(argv, add_args)), "loop"
   ```
3. One line in `main.py`'s `EXPERIMENTS`: `"<name>": "variants.deeptex.<name>.hooks"`.
4. The config: copy an existing one (`fixed/esrgan_scaled.yaml`) and give it its
   own `sub_name` and `checkpoint_root`. Keep those keys top-level, one per line:
   the ps1 files read them with line regexes.
5. `run.ps1`: copy `fixed/run.ps1` (it dot-sources `util/lib.ps1`) and change
   the trainer call to `--experiment <name>`. To evaluate with `eval_arms.py` or
   the render eval, add the tag to `Get-ArmInfo` in `util/lib.ps1`.

A variant of an existing arm that only changes numbers (another `w_render`) is
just a new config; one that needs more hooks subclasses `RenderHooks` or
`FixedHooks`.

### Experiments (exp2/exp3 style): a new arm

The config decides the arm: `esrgan_map_degradation/core/assemble.py` holds the
predicates (generator by `model.*`, discriminators by `disc.mode`, loss terms by
`loss.*`). `Criterion` is a list of terms; a term has a `key` and
`in_psnr_stage` and returns `(weight, value)` or `None`.

1. The term or module in `arms/<arm>.py` of the experiment it belongs to.
2. One condition in `assemble.py`, read with `.get(key, default)` so that every
   existing config (which lacks the key) builds exactly what it built before.
3. A config in `configs/` that inherits its reference with `base:` and sets only
   the new factor; add it to `EXPECT` in `gates/gate_configs.py` (arm ->
   reference, the keys allowed to differ), which checks that nothing else leaks
   in through inheritance.
4. A new experiment number gets `experiments/exp6_<name>/` with `arms/` and
   `configs/`, training on exp2's `core/train.py` as exp3 does.

### Data only (another MatSynth map mix)

A row in `RECIPES` (`deeptex/matsynth/recipes.py`) and a config yaml; train with
`deeptex/paper/run.ps1 -Stage paper -PaperConfig <yaml> -PaperRoot runs/deeptex/<arm>`.

### Before running it

- Do not edit `paper_loop/loop.py` except to add a hook call that does not
  exist yet; then add the default to `paper_loop/hooks.py` and regenerate
  `upstream.diff` (`python variants/tests/l0_skeleton.py --write`).
  `python variants/tests/l0_skeleton.py` checks that loop.py is still upstream's
  `train_esrgan.py` plus hook calls - the one check that stays, since it
  compares against the submodule, not the old code.
- Shared files (`hooks_common/`, `paper_loop/hooks.py`, `assemble.py`,
  `arms/base.py`) change every arm. Put the new behaviour in a new file and the
  switch in the shared one, and check that an existing arm is unchanged: run its
  short smoke (same `--seed`, `--max-steps`) before and after the change and
  compare the logged numbers. For a bit-exact comparison, the determinism
  harness (`variants/tests/_det_run.py`, `l1_trainer.py`) is at tag
  `variants-verified`.
- Smoke the new arm: DeepTextures `--max-steps 2 --save-steps 1 --no-val`
  (the term's own log line appears, a checkpoint is written); experiments,
  `gates/gate_train_smoke.py` trains every yaml in the exp2/exp3 `configs/`
  folders for a few iterations - a new experiment folder has to be added to
  its `CONFIGS` glob.

## Layout

| folder | contents |
|---|---|
| `util/` | `paths.py`, moved helpers (`map_encode`, `matsynth_extract`, `matup_lights`, `texsr_metrics`), `lib.ps1`, `preflight.ps1` |
| `deeptex/paper_loop/`, `hooks_common/`, `fixed/`, `render_loss/` (trainer, gates) | DeepTextures trainer = the paper loop plus per-experiment hooks |
| `deeptex/upstream_runner/`, `infer/`, `paper/`, `matsynth/`, `tiling/`, `render_loss/eval/`, `render_loss/build/`, `compare/` | data prep, inference, evaluation, reports |
| `matup/baseline/` | MatUp baseline |
| `experiments/` | experiments 1-5 |
| `tests/` | `l0_skeleton.py` (paper loop = upstream + hook calls); the equivalence tests are at tag `variants-verified` |
