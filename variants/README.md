# variants/

All of the project's code: the two super-resolution methods compared on
MatSynth SVBRDFs (DeepTextures, a TF ESRGAN, and MatUp, a per-material
optimisation), their data preparation, training, inference and evaluation, and
experiments 1-5. Everything runs on the unmodified submodules in `external/`
(DeepTextures `e4ee562`, matup_egsr `c74ec99`, esrgan-tf2), one folder per
experiment or shared job.

## Folders

| folder | purpose | main contents |
|---|---|---|
| `util/` | helpers no single method owns | `paths.py` (submodule paths, the only `sys.path` setup), `arms.py` (the arm registry), `arm_output.py` (writing an SR arm's evaluation folder), MatSynth extraction, map encoding, MatUp light tables, shared metrics, `lib.ps1` / `preflight.ps1` for every `run.ps1` |
| `util/report/` | comparison reports for any experiment | image I/O and conversions, metrics, sheets (strip / grid / table), the comparison-set folder layout, montages and overviews, colour drift |
| `arms.json` | the arm registry | every SR arm: method, display name, reference arm, config, checkpoint root; the head-to-head pairs |
| `matup/baseline/` | the MatUp baseline over the extracted materials | `run.ps1` (MatUp in its container), `matup_render.py`, `matup_height.py`, `matup_outputs.py` (reading a MatUp run), `postprocess.py` / `montage.py` (its comparison sets, montages and `runs/montages/`) |
| `deeptex/paper_loop/` | the DeepTextures trainer skeleton | `loop.py` = upstream `train_esrgan.py` + hook calls, `hooks.py` (defaults = the paper), `main.py --experiment paper\|fixed\|render` |
| `deeptex/hooks_common/` | trainer machinery every DeepTextures experiment shares | checkpoints (verified resume, pretrain lookup), `[metrics]` logging, NaN / exit-code handling, seeding, validation |
| `deeptex/paper/` | DeepTextures as released, on textures_db | data prep (normalize, pairs), `run.ps1` stages prep / smoke / paper / eval, the DeepTextures column and montage |
| `deeptex/upstream_runner/` | runs the submodule's own trainer; builds and checks its TFRecords | `run_upstream_train.py`, `texsr_build_tfrecord.py`, `texsr_verify_tfrecord.py` |
| `deeptex/matsynth/` | DeepTextures trained on MatSynth (all maps / channel ratio / basecolor only) | pair builder, `recipes.py` (map mix), `eval_arms.py`, arms report, colour-drift check, `run.ps1` |
| `deeptex/fixed/` | DeepTextures with the upstream defects corrected (#2-#5, #8) | `hooks.py`, corrected RaGAN, configs, `run.ps1` |
| `deeptex/render_loss/` | DeepTextures + MatUp's render loss (arms R0a / R0b / R1) | `hooks.py`, `terms.py`, `render_tf.py`, grouped dataset, `gates/`, `build/`, `eval/`, `run.ps1` |
| `deeptex/infer/` | DeepTextures inference and its comparison inputs | `deeptex_sr.py`, `deeptex_compare.py` |
| `deeptex/tiling/` | tile-boundary study of DeepTextures outputs | `deeptex_tiling_report.py` |
| `experiments/` | experiments 1-5 (PyTorch, `texsr/exp` image) | `light_inconsistency/`, `esrgan_map_degradation/`, `esrgan_render_loss/`, `matup_grf_stats/`, `speed_memory/`, plus `common/`, `matup_driver/`, `gates/`; plan and commands in `experiments/EXPERIMENTS.md` |
| `tests/` | the check to run after editing the trainer skeleton | `l0_skeleton.py` |

Each folder's README says what it is for, what is in it, how it works and how
to run it, and lists its known behaviour and defects.

## Experiment → run

| experiment | run it with |
|---|---|
| DeepTextures paper arm | `deeptex/paper/run.ps1 -Stage prep\|smoke\|paper\|eval\|all` |
| DeepTextures on MatSynth | `deeptex/matsynth/run.ps1 -Stage prep-matsynth\|eval-arms`; training: `deeptex/paper/run.ps1 -Stage paper -PaperConfig variants/deeptex/matsynth/<cfg>.yaml -PaperRoot runs/deeptex/<arm>` |
| DeepTextures fixed arm | `deeptex/fixed/run.ps1 -Stage pretrain\|train` |
| Render loss (R0a / R0b / R1) | `deeptex/render_loss/run.ps1 -Stage prep-grouped\|gate-render\|render-train\|eval-render\|eval-perceptual` |
| Tiling study | `python variants/deeptex/tiling/deeptex_tiling_report.py` |
| MatUp baseline | `matup/baseline/run.ps1` |
| Experiments 1-5 | `experiments/EXPERIMENTS.md` (from `docker/exp`: `docker compose run --rm exp python variants/experiments/...`) |

Every `run.ps1` is run as
`powershell -NoProfile -ExecutionPolicy Bypass -File variants\<...>\run.ps1 ...`
from the repo root; the four DeepTextures ones also take `-Stage preflight`.

## How the pieces fit

```
MatSynth parquet ──util/matsynth_extract──> runs/matsynth/<m>/{SVBRDF 128², GT 512²}
   │                                              │
   │                                              ├─ matup/baseline/run.ps1 ──> matup-output-<m>/<stamp>/
   │                                              │        (+ matup_render / matup_height)
   │                                              │
   │                                              └─ matup/baseline/postprocess + montage (util/report)
   │                                                    ──> compare/<stamp>/ metrics, montages
   │                                                    ▲             ──> runs/montages/ (committed)
   │                                                    │ DeepTextures column (paper hooks)
   └─ deeptex/*/ data prep ──> runs/deeptex/datasets/*.tfrecord
                                 └─ train: upstream_runner (paper) or paper_loop/main.py --experiment fixed|render
                                      └─ checkpoints ──> infer/ ──> eval (eval_arms, render eval) ──> reports

experiments/ (separate stack): /data (MatSynth, read-only) ──> /expdata/{materials,train,runs/exp1..exp5}
```

## Conventions

1. **Self-contained.** The only outside code is `external/*`. Nothing is
   imported from outside `variants/` and the submodules.
2. **Package imports, one bootstrap.** Entry scripts put the repo root on
   `sys.path` and import by package:

   ```python
   import os, sys
   sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * N)))
   from variants.util import paths      # N = depth of this file below the repo root
   paths.use_deeptex()                  # or use_matup() / use_swinir()
   ```

   Libraries import siblings as `from variants.<...> import x`, never through a
   `sys.path` edit. Submodule paths, the `DEEPTEX_ROOT` override and the pin check
   live only in `util/paths.py`.
3. **Contracts that existing runs depend on.** Log prefixes (`[metrics]`,
   `[terms]`, `[render]`, `[init]`, `[*] resumed from`, `[fatal] non-finite loss`),
   exit codes 2-9, checkpoint object names, top-level config keys (the ps1 files
   read them with line regexes), exp2 `state_dict` keys, and the output folders
   (`runs/deeptex/<arm>/`, `runs/matsynth/`, `/expdata/runs/exp1..exp5/`).
   Changing one breaks existing checkpoints, eval folders or the orchestrators.
4. **Odd behaviour is named, not silently changed.** It is made explicit in the
   module and listed in that folder's README; changing it is an experiment
   decision.
5. **No folder named `data/`.** `.gitignore:6` ignores it (the local
   `experiments/data/*.py` data-prep scripts are not in the repository).
6. **One GPU job at a time.** Before any GPU step, check that no other `texsr*`
   container is running (`docker ps`); the `run.ps1` preflights refuse to start
   otherwise (`util/preflight.ps1`, docs/RUNBOOK.md).

## How an experiment is built

An experiment is one folder plus its lines in `arms.json`. Everything it does
not change - data, inference output, evaluation, reports - comes from the
shared modules; the folder holds only its config, the code it changes, a
`run.ps1` and a README.

**Arm.** One SR training condition (method + data + recipe), named by a tag
(`matsynth_all`, `render_r1`). `<tag>_<NN>k` is one checkpoint of it
(`render_r1_40k`), and its *arm output folder*
(`runs/<method>/eval/<tag>_<NN>k/<material>/<map>.png` + `sr_manifest.json`) is
what every evaluator reads as `--arm TAG=DIR`: one row or column in a report.
LR, bilinear, MatUp and GT are built-in columns, not arms. (In
`experiments/`, "arm" means one config of experiments 2/3 - `2-2_separate` -
which is not in `arms.json` and writes no arm output folder.)

```
1 data        2 training        3 arm output             4 evaluation             5 reports
MatSynth ---> checkpoints ----> <eval>/<tag>/<m>/ -----> map / render / perceptual -> montages, tables
```

| step | what the experiment does | shared modules |
|---|---|---|
| 1 data | usually nothing; a new map mix is one `RECIPES` row | `util/matsynth_extract.py`, `util/map_encode.py`, `deeptex/matsynth/recipes.py`, `matsynth_make_pairs.py` |
| 2 training | **what the experiment is about** | DeepTextures: `deeptex/paper_loop/` + `hooks_common/` (override hooks). PyTorch: `experiments/esrgan_map_degradation/core/` (`assemble.py` picks the arm from config keys) |
| 3 arm output | wrap the model as `predict(NHWC) -> NHWC` | `util/arm_output.py` (`write_arm`, `super_resolve`, `save_sr`, the manifest) |
| 4 evaluation | pass `--arm TAG=DIR` | map space `deeptex/matsynth/deeptex_arms_report.py`, render space `deeptex/render_loss/eval/svbrdf_render_eval.py`, perceptual `svbrdf_perceptual_eval.py`, colour drift `util/report/drift.py` |
| 5 reports | nothing, or compose a new sheet | `util/report/` (images, metrics, sheets, layout, montage, overview) |
| names, pairs | one line | `arms.json` via `util/arms.py` and `Get-ArmInfo` / `Get-DisplayName` |
| orchestration | chain the steps | `util/lib.ps1` (`Invoke-Step`, `Get-CompleteEval`), `util/preflight.ps1` |

Which kind of experiment it is decides where its folder goes (details in
"Adding an experiment" below):

| kind | folder | it adds |
|---|---|---|
| a DeepTextures training change | `deeptex/<name>/` | `hooks.py` (TrainerHooks subclass), config, `run.ps1`, one `EXPERIMENTS` line in `paper_loop/main.py` |
| another SR method as an arm | `<method>/<name>/` | a writer around `arm_output.write_arm`, its training code, `methods.<method>` once in `arms.json` |
| a PyTorch experiment-list arm | `experiments/<topic>/` | `arms/<arm>.py`, configs on `base:`, one condition in `assemble.py`, an `EXPECT` row in `gate_configs.py` |
| data only | none | a `RECIPES` row, a config |
| a new report or sheet | where it is used | compose `util/report` (`images.load_fit`, `metrics.full_reference`, `sheets.grid/table`, `layout`) |

### Keeping track of each experiment

| what | where |
|---|---|
| purpose, what changed, how to run, known behaviour | the folder's `README.md` (purpose / Contents / How it works / Run / Known behaviour) |
| what differs from the reference | DeepTextures: the config header's diff command; PyTorch: `EXPECT` in `gates/gate_configs.py` |
| what it is compared with | `reference` and `head_to_head` in `arms.json` |
| which checkpoint made a result | the arm output's `sr_manifest.json` (checkpoint, step, `checkpoint_file`) |
| checkpoints / arm outputs | `runs/<method>/<arm>/checkpoints/<sub_name>/` / `runs/<method>/eval/<tag>_<NN>k/` |
| results to share (committed) | `runs/montages/<report>/` |
| smoke and check outputs | `runs/variants_verify/` |

Commit an experiment's results with its config and code, so both name one
point in history.

### From start to finish

1. Pick the reference arm and the one factor to change (a kind from the table).
2. Create the folder: config and the changed code.
3. Register the arm in `arms.json` (display, reference, `head_to_head` if any).
4. Smoke it: 2-20 training steps, write the arm output, one evaluation (minutes).
5. The full training run (hours - started by hand).
6. Arm outputs per step, then the evaluations (the `run.ps1` eval stage).
7. Update `runs/montages/<report>/` and the README; commit.

### Where the structure still falls short

- The evaluators are generic but still live under `deeptex/matsynth/` and
  `deeptex/render_loss/eval/`.
- The `run.ps1` eval stages find checkpoint steps from TF `.index` files
  (`Get-CheckpointSteps`, `Invoke-CompareAtSteps`); a torch arm needs its own
  branch there.
- The three evaluators still draw some sheets and aggregate tables with their
  own code.
- `experiments/` computes its renders and metrics with `experiments/common/`;
  to put an experiment-list run next to the DeepTextures arms, export it as an
  arm output folder (`util/arm_output.py`) and register it.

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

### Another SR method as an arm (ESRGAN, SwinIR, ...)

Every evaluator reads an arm as a folder (`--arm TAG=DIR`); the method behind it
does not matter.

1. Write the folder with `util/arm_output.py` (its docstring is the contract:
   `<DIR>/<material>/<map>.png` at the GT size, OpenGL normal, 1-channel scalar
   maps, plus `sr_manifest.json`):

   ```python
   from variants.util import arm_output
   predict = lambda batch: run_model(batch)          # NHWC float32 [0,1] -> NHWC
   mats = arm_output.materials("/workspace/runs/matsynth")
   arm_output.write_arm(mats, out, arm_output.ALL_JOBS,
                        lambda lr, where: arm_output.super_resolve(predict, lr, 4, 0, 16),
                        lambda p, sr, kind: arm_output.save_sr(p, sr, kind, "mean"),
                        ("mean", arm_output.COLLAPSE["mean"]),
                        {"checkpoint": ckpt, "step": step, "sub_name": name, "scale": 4,
                         "checkpoint_file": ckpt_path})
   ```
2. Register it in `arms.json`: the method once under `methods` (display name,
   column key prefix, checkpoint kind) and the arm under `arms` (display, reference
   arm, config, checkpoint root); add arm-vs-arm comparisons to `head_to_head`.
3. Evaluate with the existing scripts, which take any registered or unregistered
   tag: `deeptex/matsynth/deeptex_arms_report.py` (map space, dataprep image),
   `deeptex/render_loss/eval/svbrdf_render_eval.py` (render space, matup image; needs
   all five maps), `svbrdf_perceptual_eval.py` (exp image), `deeptex_dt_montage.py`,
   and `util/report/overview.py --column LABEL=STEM ...` for a materials x methods
   sheet.
4. Colour drift per checkpoint: `util/report/drift.py` with the method's
   `predict` (`deeptex/matsynth/deeptex_drift.py` is the TF driver).

### Experiments (PyTorch): a new arm

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
4. A new experiment gets its own folder named after what it studies
   (`experiments/<topic>/` with `arms/` and `configs/`), training on
   `esrgan_map_degradation/core/train.py` as `esrgan_render_loss/` does; set
   `experiment:` in its config to name its output folder under `/expdata/runs/`.

### Data only (another MatSynth map mix)

A row in `RECIPES` (`deeptex/matsynth/recipes.py`) and a config yaml; train with
`deeptex/paper/run.ps1 -Stage paper -PaperConfig <yaml> -PaperRoot runs/deeptex/<arm>`.

### Before running it

- Do not edit `paper_loop/loop.py` except to add a hook call that does not
  exist yet; then add the default to `paper_loop/hooks.py` and regenerate
  `upstream.diff` (`python variants/tests/l0_skeleton.py --write`).
  `python variants/tests/l0_skeleton.py` checks that loop.py is still upstream's
  `train_esrgan.py` plus hook calls.
- Shared files (`hooks_common/`, `paper_loop/hooks.py`, `assemble.py`,
  `arms/base.py`) change every arm. Put the new behaviour in a new file and the
  switch in the shared one, and check that an existing arm is unchanged: run its
  short smoke (same `--seed`, `--max-steps`) before and after the change and
  compare the logged numbers.
- Smoke the new arm: DeepTextures `--max-steps 2 --save-steps 1 --no-val`
  (the term's own log line appears, a checkpoint is written); experiments,
  `gates/gate_train_smoke.py` trains every yaml in the
  `esrgan_map_degradation/` and `esrgan_render_loss/` `configs/` folders for a
  few iterations - a new experiment folder has to be added to its `CONFIGS` glob.
- Write smoke and check outputs under `runs/variants_verify/` (or a scratch
  folder), never into an arm's own `runs/` folder.

History: this tree replaced the old `scripts/`, `experiments/` and `configs/`
after equivalence tests against them passed; tag `variants-verified` keeps both.
