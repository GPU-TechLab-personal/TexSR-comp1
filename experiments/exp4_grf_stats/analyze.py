#!/usr/bin/env python3
"""
실험 4 분석: does the statistics term help, and does it help MORE where
experiment 1's inconsistency C is large?

    docker compose run --rm exp python experiments/exp4_grf_stats/analyze.py --set exp1

For every (S, variant) under /expdata/runs/matup/<set>/: per-material gain
over 'orig' on the protocol metrics (sign-normalised so positive = better),
the mean gain, a Wilcoxon signed-rank test against zero, and the Spearman
correlation between gain and C (same S, from experiment 1).

Output: /expdata/runs/exp4/analysis/{summary.md, gains.csv, gain_vs_C_<S>.png}
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace")
from experiments.common import paths     # noqa: E402

# (label, path into metrics, +1 if higher is better)
METRICS = [
    ("bc_psnr", ("maps", "basecolor", "psnr"), +1),
    ("bc_lpips", ("maps", "basecolor", "lpips"), -1),
    ("bc_dE", ("maps", "basecolor", "deltaE00"), -1),
    ("n_angle", ("maps", "normal", "angle_mean"), -1),
    ("n_within5", ("maps", "normal", "within5"), +1),
    ("rough_mae", ("maps", "roughness", "mae"), -1),
    ("metal_mae", ("maps", "metallic", "mae"), -1),
    ("render_flip", ("render", "flip"), -1),
    ("render_lpips", ("render", "lpips"), -1),
]


def get(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
        if d is None:
            return np.nan
    return float(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="exp1")
    ap.add_argument("--out", default=os.path.join(paths.RUNS, "exp4", "analysis"))
    args = ap.parse_args()
    from scipy import stats
    os.makedirs(args.out, exist_ok=True)

    runs = {}
    for f in glob.glob(os.path.join(paths.RUNS, "matup", args.set, "*", "*", "*", "result.json")):
        r = json.load(open(f))
        runs[(r["upscaler"]["tag"], r["material"], r["variant"])] = r
    C = {}
    for f in glob.glob(os.path.join(paths.RUNS, "exp1", "C", "*", "*.json")):
        r = json.load(open(f))
        C[(r["upscaler"]["tag"], r["material"])] = r["C"]

    rows = []
    for (s, mat, var), r in runs.items():
        if var in ("orig", "init"):
            continue
        base = runs.get((s, mat, "orig"))
        if base is None:
            continue
        row = {"S": s, "variant": var, "material": mat, "C": C.get((s, mat), np.nan)}
        for label, path, sign in METRICS:
            row["gain_" + label] = sign * (get(r["metrics"], path) - get(base["metrics"], path))
        rows.append(row)
    if not rows:
        raise SystemExit("no variant runs with an 'orig' baseline under set %s" % args.set)

    md = ["# 실험 4 - gain of the statistics term over MatUp (set `%s`)\n" % args.set,
          "Gain = improvement over `orig` (positive = better for every metric). "
          "p = Wilcoxon vs 0; ρ(C) = Spearman of gain vs experiment-1 C, same S.\n"]
    for s in sorted({r["S"] for r in rows}):
        for var in sorted({r["variant"] for r in rows if r["S"] == s}):
            sub = [r for r in rows if r["S"] == s and r["variant"] == var]
            md.append("## S = %s, variant %s  (n=%d)\n" % (s, var, len(sub)))
            md.append("| metric | mean gain | p | ρ(gain, C) |")
            md.append("|---|---|---|---|")
            for label, _, _ in METRICS:
                g = np.array([r["gain_" + label] for r in sub], float)
                c = np.array([r["C"] for r in sub], float)
                ok = np.isfinite(g)
                p = stats.wilcoxon(g[ok]).pvalue if ok.sum() >= 6 and np.any(g[ok] != 0) else np.nan
                okc = ok & np.isfinite(c)
                rho = stats.spearmanr(c[okc], g[okc])[0] if okc.sum() >= 4 else np.nan
                md.append("| %s | %+.5f | %s | %s |" % (label, np.nanmean(g),
                                                        "—" if np.isnan(p) else "%.3f" % p,
                                                        "—" if np.isnan(rho) else "%+.2f" % rho))
            md.append("")
            plot(sub, s, var, args.out)
    with open(os.path.join(args.out, "gains.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))
    print("-> " + args.out)


def plot(sub, s, var, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["gain_render_flip", "gain_n_angle", "gain_bc_lpips"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6))
    c = [r["C"] for r in sub]
    for ax, k in zip(axes, keys):
        ax.scatter(c, [r[k] for r in sub], s=14)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("C (experiment 1)")
        ax.set_ylabel(k)
    fig.suptitle("S=%s  variant=%s" % (s, var))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gain_vs_C_%s_%s.png" % (s, var)), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
