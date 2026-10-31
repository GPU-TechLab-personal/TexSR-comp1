#!/usr/bin/env python3
"""Collect the experiments verification into <out>/report.md.

    python variants/tests/l6_experiments_report.py --out runs/variants_verify/experiments/<stamp>

Reads <out>/l0.txt (l0_static.py stdout, experiments.json only), <out>/imports.txt,
<out>/cpu.json, <out>/gpu.json - whatever exists - and writes one table per layer
with PASS/FAIL and the measured differences.
"""
import argparse
import json
import os
import subprocess

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))


def git(*a):
    try:
        return subprocess.check_output(["git", "-C", REPO] + list(a), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "?"


def esc(s):
    return str(s).replace("|", "\\|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    o = args.out
    md = ["# variants/experiments verification", "",
          "- branch `%s`, commit `%s`" % (git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "--short", "HEAD")),
          "- folder `%s`" % os.path.abspath(o).replace(os.sep, "/"), ""]
    summary = []

    l0 = os.path.join(o, "l0.txt")
    if os.path.isfile(l0):
        lines = open(l0, encoding="utf-8").read().splitlines()
        rows = [l for l in lines if l.startswith(("PASS", "FAIL"))]
        last = next((l for l in reversed(lines) if l.startswith("L0 static")), "")
        n_fail = sum(l.startswith("FAIL") for l in rows)
        summary.append(("L0 static (moves/experiments.json)", len(rows), n_fail, last))
        full = os.path.join(o, "l0_all.txt")
        if os.path.isfile(full):
            fl = open(full, encoding="utf-8").read().splitlines()
            fr = [l for l in fl if l.startswith(("PASS", "FAIL"))]
            summary.append(("L0 static (every moves/*.json)", len(fr), sum(l.startswith("FAIL") for l in fr),
                            next((l for l in reversed(fl) if l.startswith("L0 static")), "")))
        md += ["## L0 - static (AST identity / line coverage)", "", "```", *lines, "```", ""]

    imp = os.path.join(o, "imports.txt")
    if os.path.isfile(imp):
        lines = open(imp, encoding="utf-8").read().splitlines()
        rows = [l for l in lines if l.startswith(("PASS", "FAIL"))]
        n_fail = sum(l.startswith("FAIL") for l in rows)
        summary.append(("L6 imports (no scripts/ or experiments/)", len(rows), n_fail, ""))
        md += ["## L6 imports", "", "%d modules, %d FAIL. Every module of variants/experiments imports in a fresh "
               "interpreter and pulls in no `experiments.*`, `scripts.*` or `texsr_metrics` from scripts/." % (
                   len(rows), n_fail), ""]
        md += ["<details><summary>modules</summary>", "", "```", *rows, "```", "</details>", ""]

    for fname, title in (("cpu.json", "L6 CPU - exact equality (torch.equal / ==)"),
                         ("gpu.json", "L6 GPU")):
        p = os.path.join(o, fname)
        if not os.path.isfile(p):
            continue
        d = json.load(open(p))
        res = d["results"]
        head = ", ".join("%s %s" % (k, d[k]) for k in ("torch", "gpu", "threads") if k in d)
        md += ["## %s" % title, "", head + ("; %.0f s" % d["seconds"] if "seconds" in d else ""), ""]
        if "old_model_block_lines" in d:
            md += ["Old trainer side of the arm checks = `%s`, exec'd from the old source." % d["old_model_block_lines"], ""]
        for layer in dict.fromkeys(r["layer"] for r in res):
            rr = [r for r in res if r["layer"] == layer]
            nf = sum(not r["pass"] for r in rr)
            summary.append(("L6 %s %s" % (fname.split(".")[0], layer), len(rr), nf, ""))
            md += ["### %s (%d checks, %d FAIL)" % (layer, len(rr), nf), "", "| verdict | check | detail |",
                   "|---|---|---|"]
            md += ["| %s | %s | %s |" % ("PASS" if r["pass"] else "**FAIL**", esc(r["check"]), esc(r["detail"]))
                   for r in rr]
            md.append("")
            for r in rr:
                if "max_abs_new_vs_old" in r:
                    md += ["`%s`: allclose(rtol=0, atol=%g). Measured max |new - old| per map: %s. "
                           "Old vs old (same code, run twice): %s." % (
                               r["check"], r["atol"],
                               ", ".join("%s %.3e" % kv for kv in r["max_abs_new_vs_old"].items()),
                               ", ".join("%s %.3e" % kv for kv in r["max_abs_old_vs_old"].items())), ""]

    md[5:5] = ["## Summary", "", "| layer | checks | FAIL | |", "|---|---|---|---|"] + [
        "| %s | %d | %d | %s |" % (a, b, c, esc(e)) for a, b, c, e in summary] + [
        "", "**%s**" % ("ALL PASS" if all(c == 0 for _, _, c, _ in summary) else "FAILURES"), ""]
    with open(os.path.join(o, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[:5 + 6 + len(summary)]))
    return 0 if all(c == 0 for _, _, c, _ in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
