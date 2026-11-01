#!/usr/bin/env python3
"""KAIR's own SwinIR training scripts, run unmodified, on swinir_{cls,real}_ft.json.

Runs in the srft service:

    docker compose run --rm srft python variants/sr_finetune/swinir/kair_train.py \\
        --opt variants/sr_finetune/swinir/swinir_real_ft.json

external/KAIR/main_train_psnr.py ("model": "plain") or main_train_gan.py ("gan") is
executed with runpy as __main__ - the official loop, losses, EMA, schedulers and
saving. This file adds three things around it, each for a KAIR behaviour named in
variants/sr_finetune/README.md:

1. STARTING WEIGHTS. KAIR ignores the json's pretrained_netG: main_train_*.py call
   find_last_checkpoint() without pretrained_path, so an empty models/ folder means
   random weights. Before the first run this writes models/0_G.pth and 0_E.pth from
   the official release (weights.kair_init; step 0, so the schedule starts at the
   beginning). A folder that already holds a checkpoint is a resume and is left alone.

2. AN END. The loop is `for epoch in range(1000000)` with no total; it only ever
   saves. models.select_model.define_Model is wrapped so that the model it returns
   saves as usual and then, at step >= --total-iter, exits 0. That wrapper is the
   only change to the official run; --total-iter must be a multiple of the json's
   checkpoint_save, so the last step is always saved.

3. OVERRIDES without a second json. --set train.checkpoint_save=100 (dotted keys,
   JSON values; a value that is not JSON is taken as a string) edits the loaded options; the gates shorten runs this way. The
   options actually run are written to <root>/_opt/<task>.json and passed to KAIR
   (which also copies them into <task>/options/).

Resume: run the same command again - KAIR continues from the highest <N>_G/E and
<N>_optimizerG in models/ (and, for the GAN, <N>_D; D's optimizer state is not kept
by the official D_optimizer_reuse: false). The learning rate is recomputed from the
step (MultiStepLR.step(n), models/model_base.py:60-62), so the schedule continues.
"""
import argparse
import json
import os
import runpy
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.util import paths  # noqa: E402
from variants.sr_finetune import weights  # noqa: E402

SCRIPTS = {"plain": "main_train_psnr.py", "gan": "main_train_gan.py"}
INIT = {"swinir_cls_ft": "swinir_cls", "swinir_real_ft": "swinir_real"}


def load_options(path):
    """KAIR's own comment rule (utils/utils_option.py parse): drop everything after //."""
    with open(path) as f:
        return json.loads("".join(line.split("//")[0] + "\n" for line in f))


def apply_set(opt, items):
    for item in items:
        key, value = item.split("=", 1)
        node, parts = opt, key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise SystemExit("[fatal] --set %s: no such key (only existing keys may be overridden)" % key)
        try:
            node[parts[-1]] = json.loads(value)
        except ValueError:          # a bare string: PowerShell 5.1 strips quotes from native arguments
            node[parts[-1]] = value


def stop_after(total):
    """Wrap define_Model: the model's save() exits 0 once step >= total."""
    import models.select_model as select_model
    official = select_model.define_Model

    def define_Model(opt):
        model = official(opt)
        save = model.save

        def save_then_stop(iter_label):
            save(iter_label)
            if int(iter_label) >= total:
                print("[kair_train] saved step %d >= --total-iter %d: done" % (int(iter_label), total), flush=True)
                sys.exit(0)
        model.save = save_then_stop
        return model
    select_model.define_Model = define_Model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opt", required=True, help="swinir_cls_ft.json or swinir_real_ft.json")
    ap.add_argument("--total-iter", type=int, default=40000)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    args = ap.parse_args()

    opt = load_options(args.opt)
    apply_set(opt, args.set)
    task, root = opt["task"], opt["path"]["root"]
    if not os.path.isabs(root) or os.path.abspath(root).startswith(paths.EXTERNAL):
        raise SystemExit("[fatal] path.root must be absolute and outside external/: %s" % root)
    save_every = opt["train"]["checkpoint_save"]
    if args.total_iter % save_every:
        raise SystemExit("[fatal] --total-iter %d is not a multiple of checkpoint_save %d" % (args.total_iter, save_every))
    if opt["train"]["checkpoint_test"] <= args.total_iter:
        raise SystemExit("[fatal] checkpoint_test %d would be reached, and there is no test set"
                         % opt["train"]["checkpoint_test"])

    models_dir = os.path.join(root, task, "models")
    if not any(f.endswith(("_G.pth", "_E.pth")) for f in (os.listdir(models_dir) if os.path.isdir(models_dir) else [])):
        arm = INIT.get(task) or INIT[os.path.basename(args.opt)[:-5]]
        weights.kair_init(arm, models_dir)

    run_opt = os.path.join(root, "_opt", task + ".json")
    os.makedirs(os.path.dirname(run_opt), exist_ok=True)
    with open(run_opt, "w") as f:
        json.dump(opt, f, indent=2)

    paths.use_kair()
    stop_after(args.total_iter)
    sys.argv = [SCRIPTS[opt["model"]], "--opt", run_opt]
    runpy.run_path(os.path.join(paths.KAIR_ROOT, SCRIPTS[opt["model"]]), run_name="__main__")


if __name__ == "__main__":
    main()
