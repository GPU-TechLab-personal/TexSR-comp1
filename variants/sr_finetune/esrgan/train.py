#!/usr/bin/env python3
"""BasicSR's own training pipeline on esrgan_ft.yml - nothing of it re-implemented.

Runs in the srft service:

    docker compose run --rm srft python variants/sr_finetune/esrgan/train.py \\
        -opt variants/sr_finetune/esrgan/esrgan_ft.yml --auto_resume

Everything after this script's own --root goes to basicsr.train.train_pipeline
unchanged (-opt, --auto_resume, --force_yml key=value ...): the gates shorten a
run with --force_yml train:total_iter=300 logger:save_checkpoint_freq=100 name=...
instead of keeping a second yml.

--root  BasicSR's root_path: experiments/<name>/ (models/, training_states/, the
        log) and tb_logger/<name>/ go under it. Default /workspace/runs/sr_ft/basicsr,
        which arms.json's esrgan_ft checkpoint_root assumes.

--auto_resume continues from the newest training_states/<N>.state (optimizers,
schedulers, iteration). Without it, a second run under the same name renames the
first experiment folder to <name>_archived_<time> (BasicSR's make_exp_dirs) - never
delete one by accident; run.ps1 always passes --auto_resume.

CWD = ROOT. BasicSR saves under root_path, but load_resume_state looks for
osp.join('experiments', name, 'training_states') RELATIVE TO THE WORKING DIRECTORY
(basicsr/train.py:71). From /workspace that path does not exist, so --auto_resume
silently starts over - and make_exp_dirs then archives the run it should have
resumed. This script chdirs to --root before handing over (after making -opt
absolute); every other path in esrgan_ft.yml is absolute.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.util import paths  # noqa: E402

DEFAULT_ROOT = "/workspace/runs/sr_ft/basicsr"


def main():
    argv = sys.argv[1:]
    root = DEFAULT_ROOT
    if "--root" in argv:
        i = argv.index("--root")
        root = argv[i + 1]
        del argv[i:i + 2]
    if not os.path.isabs(root) or os.path.abspath(root).startswith(paths.EXTERNAL):
        raise SystemExit("[fatal] --root must be absolute and outside external/: %s" % root)
    os.makedirs(root, exist_ok=True)
    if "-opt" in argv:
        i = argv.index("-opt") + 1
        argv[i] = os.path.abspath(argv[i])

    paths.use_basicsr()
    from basicsr.train import train_pipeline
    sys.argv = [sys.argv[0]] + argv
    os.chdir(root)                      # see CWD = ROOT above
    train_pipeline(root)


if __name__ == "__main__":
    main()
