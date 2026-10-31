#!/usr/bin/env python3
"""
실험 4: MatUp original vs + statistics term (variant A: per-map, B: + cross-map).

    # pilot, one lambda
    docker compose run --rm exp python experiments/exp4_grf_stats/run.py --pilot --s swinir
    # full: 50 materials, the same S as experiment 1, a lambda sweep
    docker compose run --rm exp python experiments/exp4_grf_stats/run.py --s swinir sdx4 --lambdas 0.003 0.01 0.03

Per (material, S): 'orig' (shared with experiment 1 - reused if present, with
residuals) and one run per (variant, lambda), tagged e.g. "A_l0.01". Each is a
full MatUp optimisation (~1-3 min). Estimated targets (Sigma, Sigma_g, noise)
are stored in result.json under extra_info.targets.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/workspace")
from experiments.common import cli                                   # noqa: E402
from experiments.common.matup_runs import run_one, run_dir            # noqa: E402
from experiments.exp4_grf_stats.grf_stats import factory             # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_material_args(ap)
    cli.add_s_args(ap)
    ap.add_argument("--variants", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.003])
    ap.add_argument("--lambda-grad", type=float, default=1.0)
    ap.add_argument("--stat-exclude", nargs="*", default=[], choices=["basecolor", "metallic", "normal", "roughness"],
                    help="maps left out of the statistics (see grf_stats.structure_mask)")
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--n-patches", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dev = cli.device()
    names = cli.resolve_materials(args)
    for up in cli.iter_upscalers(args, dev):
        for name in names:
            run_one(args.set, name, up, dev, variant="orig", epochs=args.epochs,
                    residuals=True, force=False)
            for v in args.variants:
                for lam in args.lambdas:
                    tag = "%s_l%g" % (v, lam) + ("" if args.lambda_grad == 1.0 else "_g%g" % args.lambda_grad)
                    if args.stat_exclude:
                        tag += "_no-" + "-".join(sorted(args.stat_exclude))
                    f = factory(v, lam, args.lambda_grad, patch=args.patch, n_patches=args.n_patches,
                                exclude=tuple(args.stat_exclude))
                    run_one(args.set, name, up, dev, variant=tag, extra_factory=f,
                            epochs=args.epochs, force=args.force,
                            extra_info={"variant": v, "lambda": lam, "lambda_grad": args.lambda_grad,
                                        "patch": args.patch, "n_patches": args.n_patches,
                                        "stat_exclude": args.stat_exclude})
                    if f.last_info is not None:          # freshly optimised: attach the targets
                        p = os.path.join(run_dir(args.set, up.tag, name, tag), "result.json")
                        r = json.load(open(p))
                        r.setdefault("extra_info", {})["targets"] = f.last_info
                        json.dump(r, open(p, "w"), indent=2)


if __name__ == "__main__":
    main()
