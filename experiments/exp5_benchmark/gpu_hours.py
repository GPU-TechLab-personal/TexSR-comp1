#!/usr/bin/env python3
"""
실험 5: GPU-hours until a training run first reaches a target metric.

    docker compose run --rm exp python experiments/exp5_benchmark/gpu_hours.py \
        --target normal_angle<=3.0 --target basecolor_psnr>=28

Reads every /expdata/runs/{exp2,exp3}/*/eval.jsonl written by train.py.
train_seconds there counts training steps only (evaluation and checkpoint
time excluded, CUDA-synchronised), so hours = train_seconds / 3600 at the
first evaluation meeting the target. "not reached" is reported as such, with
the best value seen - never extrapolated.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, "/workspace")
from experiments.common import paths    # noqa: E402


def parse(t):
    m = re.fullmatch(r"(\w+)\s*(<=|>=)\s*([-\d.eE]+)", t)
    if not m:
        raise SystemExit("target must look like metric<=value or metric>=value: " + t)
    return m.group(1), m.group(2), float(m.group(3))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", required=True)
    ap.add_argument("--runs", nargs="*", default=["exp2", "exp3"])
    args = ap.parse_args()
    targets = [parse(t) for t in args.target]
    rows = []
    for exp in args.runs:
        for f in sorted(glob.glob(os.path.join(paths.RUNS, exp, "*", "eval.jsonl"))):
            ev = [json.loads(l) for l in open(f) if l.strip()]
            run = "%s/%s" % (exp, os.path.basename(os.path.dirname(f)))
            for metric, op, val in targets:
                hit = next((e for e in ev if metric in e and
                            (e[metric] <= val if op == "<=" else e[metric] >= val)), None)
                best = (min if op == "<=" else max)((e[metric] for e in ev if metric in e), default=None)
                rows.append({"run": run, "target": "%s%s%g" % (metric, op, val),
                             "iter": hit["iter"] if hit else None,
                             "gpu_hours": hit["train_seconds"] / 3600 if hit else None,
                             "best": best, "evals": len(ev)})
    print("| run | target | reached at iter | GPU-hours | best seen |")
    print("|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %s | %s | %s |" % (
            r["run"], r["target"], r["iter"] or "not reached",
            "%.2f" % r["gpu_hours"] if r["gpu_hours"] is not None else "—",
            "%.4f" % r["best"] if r["best"] is not None else "—"))
    out = os.path.join(paths.RUNS, "exp5", "gpu_hours.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
