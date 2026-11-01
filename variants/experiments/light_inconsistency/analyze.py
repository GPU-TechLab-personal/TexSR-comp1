#!/usr/bin/env python3
"""
실험 1 분석: does inconsistency follow the material (normal detail, roughness)
or the upsampler S?

    python variants/experiments/light_inconsistency/analyze.py --set exp1

For every S found under /expdata/runs/exp1/C/: Pearson and Spearman
correlation of C (and C_rel) with
    normal_grad_var, normal_tilt_deg, roughness_mean    (material hypotheses)
    basecolor_lum_std, basecolor_rms_contrast           (control)
using GT factors (LR factors in the CSV too). Reading: if C tracks the
material factors with a similar sign/strength for every S, the inconsistency
is a material property; if the S ordering dominates and factor correlations
differ by S, it is an S property. The control should correlate weakly -
if it does not, C is picking up texture contrast, not lighting inconsistency.

If MatUp residual runs exist (run_residuals.py), the discard fractions are
correlated with the same factors.

Output: /expdata/runs/exp1/analysis/{summary.md, correlations.csv, scatter_<S>.png}
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.experiments.common import paths     # noqa: E402

FACTORS = ["normal_grad_var", "normal_tilt_deg", "roughness_mean",
           "basecolor_lum_std", "basecolor_rms_contrast"]
CONTROL = {"basecolor_lum_std", "basecolor_rms_contrast"}


def corr(x, y):
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None, None, None, None, int(ok.sum())
    p = stats.pearsonr(x[ok], y[ok])
    s = stats.spearmanr(x[ok], y[ok])
    return float(p[0]), float(p[1]), float(s[0]), float(s[1]), int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="exp1")
    ap.add_argument("--out", default=os.path.join(paths.RUNS, "exp1", "analysis"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    by_s = {}
    for f in sorted(glob.glob(os.path.join(paths.RUNS, "exp1", "C", "*", "*.json"))):
        r = json.load(open(f))
        if r.get("set") == args.set:
            by_s.setdefault(r["upscaler"]["tag"], []).append(r)
    if not by_s:
        raise SystemExit("no C results for set %s - run run_C.py first" % args.set)

    rows, md = [], ["# 실험 1 - C vs material factors (set `%s`)\n" % args.set]
    md.append("| S | n | mean C | mean C_rel | " + " | ".join(
        "ρ(C, %s)%s" % (f, " (ctrl)" if f in CONTROL else "") for f in FACTORS) + " |")
    md.append("|---|---|---|---|" + "---|" * len(FACTORS))
    for tag, recs in sorted(by_s.items()):
        cells = []
        for target in ("C", "C_rel"):
            y = [r[target] if r[target] is not None else np.nan for r in recs]
            for src in ("factors_gt", "factors_lr"):
                for f in FACTORS:
                    pr, pp, sr, sp, n = corr([r[src][f] for r in recs], y)
                    rows.append({"S": tag, "target": target, "factors": src, "factor": f,
                                 "n": n, "pearson": pr, "pearson_p": pp,
                                 "spearman": sr, "spearman_p": sp})
                    if target == "C" and src == "factors_gt":
                        cells.append("—" if sr is None else "%+.2f%s" % (sr, "*" if sp < 0.05 else ""))
        md.append("| %s | %d | %.5f | %.3f | %s |" % (
            tag, len(recs), np.mean([r["C"] for r in recs]),
            np.nanmean([r["C_rel"] or np.nan for r in recs]), " | ".join(cells)))
        scatter(recs, tag, args.out)
    md.append("\nρ = Spearman over materials, GT factors; * p < 0.05. Full table: correlations.csv.\n")

    # residuals (정밀 지표), if run_residuals.py has been run
    res = glob.glob(os.path.join(paths.RUNS, "matup", args.set, "*", "*", "orig", "result.json"))
    if res:
        md.append("## MatUp residuals after optimisation\n")
        md.append("| S | n | mean residual | >2x median | >3x median | ρ(>2x, normal_grad_var) | ρ(>2x, roughness_mean) |")
        md.append("|---|---|---|---|---|---|---|")
        by = {}
        for f in res:
            r = json.load(open(f))
            if "residuals" in r:
                by.setdefault(r["upscaler"]["tag"], []).append(r)
        c_lookup = {(r["upscaler"]["tag"], r["material"]): r for recs in by_s.values() for r in recs}
        for tag, recs in sorted(by.items()):
            fr = [r["residuals"]["discard_rel"]["gt_2x_median"] for r in recs]
            fac = [c_lookup.get((tag, r["material"]), {}).get("factors_gt") for r in recs]
            cells = []
            for f in ("normal_grad_var", "roughness_mean"):
                xs = [fa[f] if fa else np.nan for fa in fac]
                _, _, sr, sp, _ = corr(xs, fr)
                cells.append("—" if sr is None else "%+.2f%s" % (sr, "*" if sp < 0.05 else ""))
            md.append("| %s | %d | %.4f | %.3f | %.3f | %s |" % (
                tag, len(recs), np.mean([r["residuals"]["mean"] for r in recs]), np.mean(fr),
                np.mean([r["residuals"]["discard_rel"]["gt_3x_median"] for r in recs]),
                " | ".join(cells)))

    with open(os.path.join(args.out, "correlations.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\n-> " + args.out)


def scatter(recs, tag, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(FACTORS), figsize=(4 * len(FACTORS), 3.6))
    y = [r["C"] for r in recs]
    for ax, f in zip(axes, FACTORS):
        x = [r["factors_gt"][f] for r in recs]
        ax.scatter(x, y, s=14)
        if f == "normal_grad_var":
            ax.set_xscale("log")
        ax.set_xlabel(f + (" (control)" if f in CONTROL else ""))
        ax.set_ylabel("C")
    fig.suptitle("S = %s  (n=%d)" % (tag, len(recs)))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scatter_%s.png" % tag), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
