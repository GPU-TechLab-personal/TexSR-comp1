#!/usr/bin/env python3
"""
실험 1 (정밀 지표): run MatUp, then look at the per-light residuals it leaves.

For every texel and light: r_w(x) = || R_w(MatUp maps)(x) - S(R_w)(x) ||_2.
What l2,1 "threw away" is read off that distribution: the share of
(texel, light) pairs far above the texel's median residual, absolute-threshold
shares, per-light means and a log histogram (matup_driver.per_light_residuals).

    python variants/experiments/light_inconsistency/run_residuals.py --pilot --s swinir
    python variants/experiments/light_inconsistency/run_residuals.py --s swinir sdx4 --with-init

One MatUp optimisation per (material, S): ~1-3 min on the 3080 Ti. The run is
stored as the 'orig' variant under /expdata/runs/matup/, which experiment 4
reuses as its baseline instead of recomputing it.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.experiments.common import cli                   # noqa: E402
from variants.experiments.common.matup_runs import run_one     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_material_args(ap)
    cli.add_s_args(ap)
    ap.add_argument("--epochs", type=int, default=10, help="MatUp default 10")
    ap.add_argument("--with-init", action="store_true",
                    help="also score the bilinear initialisation (no optimisation)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dev = cli.device()
    names = cli.resolve_materials(args)
    for up in cli.iter_upscalers(args, dev):
        for name in names:
            if args.with_init:
                run_one(args.set, name, up, dev, variant="init", force=args.force)
            r = run_one(args.set, name, up, dev, variant="orig", epochs=args.epochs,
                        residuals=True, force=args.force)
            if "residuals" not in r:
                # cached from a run that did not ask for residuals (e.g. exp4)
                r = run_one(args.set, name, up, dev, variant="orig", epochs=args.epochs,
                            residuals=True, force=True)
            res = r["residuals"]
            print("      residual mean %.4f  >2x median %.3f  >3x median %.3f" % (
                res["mean"], res["discard_rel"]["gt_2x_median"], res["discard_rel"]["gt_3x_median"]))


if __name__ == "__main__":
    main()
