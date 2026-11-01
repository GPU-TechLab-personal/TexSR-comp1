#!/usr/bin/env python3
"""DeepTextures trainer = the paper loop + one experiment's hooks.

    python variants/deeptex/paper_loop/main.py --experiment paper  --cfg_path X [--gpu 0]
    python variants/deeptex/paper_loop/main.py --experiment fixed  --stage psnr|esrgan ...
    python variants/deeptex/paper_loop/main.py --experiment render ...

paper   loop.py with paper_loop/hooks.py's defaults: train_esrgan.py's behaviour,
        line for line (L1a). The PAPER ARM IS NOT TRAINED THIS WAY - it runs the
        submodule's own train_esrgan.py through upstream_runner/; this entry
        exists to prove that loop.py + default hooks == train_esrgan.py.
fixed   fixed/hooks.py       the old `deeptex_train.py` on a non-grouped config
render  render_loss/hooks.py the old `deeptex_train.py` on a grouped config

Every flag after --experiment is the old deeptex_train.py's, unchanged, so a
run.ps1 passes the argv it always passed (L5).
"""

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))
sys.path.insert(0, REPO)

EXPERIMENTS = {
    "fixed": "variants.deeptex.fixed.hooks",
    "render": "variants.deeptex.render_loss.hooks",
}


def split_experiment(argv):
    if "--experiment" not in argv:
        sys.exit("main.py: --experiment {paper,%s} is required" % ",".join(EXPERIMENTS))
    i = argv.index("--experiment")
    if i + 1 >= len(argv):
        sys.exit("main.py: --experiment needs a value")
    name = argv[i + 1]
    if name != "paper" and name not in EXPERIMENTS:
        sys.exit("main.py: unknown experiment %r (paper, %s)" % (name, ", ".join(EXPERIMENTS)))
    return name, argv[:i] + argv[i + 2:]


def main():
    name, rest = split_experiment(sys.argv[1:])
    if name != "paper":
        # Before importing tensorflow: the C++ layer reads this at import time,
        # which is why train_esrgan.py:20 setting it after the :4 import does
        # nothing. 2 = hide INFO, keep WARNING and ERROR (defect #4).
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    from variants.util import paths
    paths.use_deeptex()

    if name == "paper":
        from absl import app
        from variants.deeptex.paper_loop import loop
        app.run(loop.main, argv=[sys.argv[0]] + rest)
        return

    import importlib
    from absl.flags import FLAGS
    hooks, loop_name = importlib.import_module(EXPERIMENTS[name]).build(rest)
    loop = importlib.import_module("variants.deeptex.paper_loop." + loop_name)
    # The loop reads --cfg_path / --gpu from absl FLAGS, as upstream does.
    FLAGS([sys.argv[0], "--cfg_path", hooks.args.cfg_path, "--gpu", hooks.args.gpu])
    loop.main(None, hooks=hooks)


if __name__ == "__main__":
    main()
