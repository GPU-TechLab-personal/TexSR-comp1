"""MatUp's optimisation, driven from outside with an optional extra loss term.

Split from experiments/common/matup_driver.py by concern; every function is the
old one unchanged (L0 "functions" entries). Shared by experiment 1
(run_residuals via common/matup_runs.py), 4 (the stat-term variants) and 5
(bench_train's MatUp step).

    upstream.py    the MatUp modules reused verbatim
    args.py        matup_args             (i)
    optimise.py    optimise               (a)-(h), (k); the extra_loss hook (f)
    reference.py   init_maps              (j)
    outputs.py     write_outputs (e), to_canonical, save_json
    residuals.py   per_light_residuals    (experiment 1, an addition)

The changes (a)-(k) against the submodule are tabled in README.md. The old
module's names stay reachable here, so `from variants.experiments import
matup_driver as D; D.optimise(...)`, `D.Renderer`, `D.mtrain` ... work as before.
"""
from variants.experiments.matup_driver.upstream import (FullyConnected, LIGHT_RADIUS,  # noqa: F401
                                                        MatUpDataset, Renderer, data_utils,
                                                        light_utils, mtrain, render_utils, testing)
from variants.experiments.matup_driver.args import matup_args                        # noqa: F401
from variants.experiments.matup_driver.optimise import optimise                      # noqa: F401
from variants.experiments.matup_driver.reference import init_maps                    # noqa: F401
from variants.experiments.matup_driver.outputs import save_json, to_canonical, write_outputs  # noqa: F401
from variants.experiments.matup_driver.residuals import per_light_residuals          # noqa: F401
