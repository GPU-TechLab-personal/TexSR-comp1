#!/usr/bin/env python3
"""Gate registry: adding the SR arms to variants/arms.json changed nothing any
existing report computes, and the new entries are consistent. Host, stdlib only:

    python variants/sr_finetune/gates/gate_registry.py --snapshot   # BEFORE editing arms.json
    python variants/sr_finetune/gates/gate_registry.py              # after

--snapshot writes runs/sr_ft/gate/registry_before.json: for the arms already in
arms.json (every base tag, bare and with the _30k/_35k/_40k suffixes, plus one
unregistered tag), everything util/arms.py answers about them -

  per tag     display_name, base_display, label_lines, column_key, method,
              method_display, reference, and lib.ps1's Get-DisplayName
  per subset  first_pair over every 2-, 3- and 4-subset of the bare base tags
              (what the map report picks), pairs() over every 2- and 3-subset of
              the tags at one step and over all of them at once (what the render
              eval pairs)

The default run computes the same from the edited arms.json and requires it to
be IDENTICAL (unchanged), then checks the new entries (new):

  - nets.ARMS and the arms.json entries of the new methods name the same tags
  - every new method is checkpoint "torch"; its key prefix is not a fixed column
    kind (bilinear, matup - svbrdf_render_eval.py treats those specially) and is
    not used by another method
  - every reference and every head_to_head tag is a registered arm
  - every new arm's display name and Get-DisplayName agree, bare and at 40k

and runs variants/tests/l0_skeleton.py (skeleton). -> runs/sr_ft/gate/registry.json
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.util import arms, paths  # noqa: E402

GATE_DIR = os.path.join(paths.REPO, "runs", "sr_ft", "gate")
BEFORE = os.path.join(GATE_DIR, "registry_before.json")
NEW_METHODS = ("esrgan_bsr", "swinir_kair")
FIXED_KINDS = ("bilinear", "matup")
STEPS = ("", "_30k", "_35k", "_40k")


def ps_display(tags):
    """lib.ps1's Get-DisplayName for each tag (one PowerShell process)."""
    lib = os.path.join(paths.VARIANTS, "util", "lib.ps1")
    script = ("$repo = '%s'; . '%s'; $o = [ordered]@{}; foreach ($t in @(%s)) { $o[$t] = Get-DisplayName $t }; "
              "$o | ConvertTo-Json -Compress") % (paths.REPO, lib, ",".join("'%s'" % t for t in tags))
    out = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command", script]).decode("utf-8", "replace")
    return json.loads(out)


def answers(bases):
    """Everything arms.py (and Get-DisplayName) says about these base tags."""
    tags = [b + s for b in bases for s in STEPS] + ["not_registered_40k"]
    ps = ps_display(tags)
    per_tag = {t: {"display_name": arms.display_name(t), "base_display": arms.base_display(t),
                   "label_lines": arms.label_lines(arms.display_name(t)), "column_key": arms.column_key(t),
                   "method": arms.method(t), "method_display": arms.method_display(t),
                   "reference": arms.reference(t), "ps_display": ps[t]} for t in tags}
    first = {",".join(c): arms.first_pair(list(c))
             for n in (2, 3, 4) for c in itertools.combinations(bases, n)}
    pr = {}
    for s in STEPS:
        at = [b + s for b in bases]
        for n in (2, 3):
            for c in itertools.combinations(at, n):
                pr[",".join(c)] = arms.pairs(list(c))
    every = [b + s for b in bases for s in STEPS]
    pr["ALL"] = arms.pairs(every)
    return json.loads(json.dumps({"bases": list(bases), "per_tag": per_tag,
                                  "first_pair": first, "pairs": pr}))


def check_new():
    reg = arms.registry()
    problems = []
    new_tags = sorted(t for t, e in reg["arms"].items() if e["method"] in NEW_METHODS)
    sys.path.insert(0, paths.REPO)
    from variants.sr_finetune import nets
    if sorted(nets.ARMS) != new_tags:
        problems.append("nets.ARMS %s != arms.json %s" % (sorted(nets.ARMS), new_tags))
    prefixes = {}
    for m, info in reg["methods"].items():
        prefixes.setdefault(info["key_prefix"], []).append(m)
    for m in NEW_METHODS:
        info = reg["methods"].get(m)
        if not info:
            problems.append("method %s missing" % m)
            continue
        if info["checkpoint"] != "torch":
            problems.append("%s: checkpoint %s, not torch" % (m, info["checkpoint"]))
        kind = info["key_prefix"].split("_")[0]
        if kind in FIXED_KINDS:
            problems.append("%s: key prefix %s is a fixed column kind" % (m, info["key_prefix"]))
        if len(prefixes[info["key_prefix"]]) > 1:
            problems.append("key prefix %s shared by %s" % (info["key_prefix"], prefixes[info["key_prefix"]]))
    for t, e in reg["arms"].items():
        if e.get("reference") and e["reference"] not in reg["arms"]:
            problems.append("%s: reference %s is not an arm" % (t, e["reference"]))
    for a, b in reg["head_to_head"]:
        for t in (a, b):
            if t not in reg["arms"]:
                problems.append("head_to_head %s-%s: %s is not an arm" % (a, b, t))
    shown = [t + s for t in new_tags for s in ("", "_40k")]
    ps = ps_display(shown)
    for t in shown:
        if ps[t] != arms.display_name(t):
            problems.append("%s: Get-DisplayName %r != display_name %r" % (t, ps[t], arms.display_name(t)))
    return {"ok": not problems, "problems": problems, "new_arms": new_tags,
            "display": {t: arms.display_name(t) for t in shown},
            "column_key": {t: arms.column_key(t) for t in new_tags},
            "new_pairs": [p for p in reg["head_to_head"] if p[0] in new_tags or p[1] in new_tags]}


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k), b.get(k), path + "/" + k)
        return out
    return [] if a == b else ["%s: %r -> %r" % (path, a, b)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", action="store_true")
    args = ap.parse_args()
    os.makedirs(GATE_DIR, exist_ok=True)
    if args.snapshot:
        bases = sorted(t for t, e in arms.registry()["arms"].items() if e["method"] not in NEW_METHODS)
        snap = answers(bases)
        with open(BEFORE, "w") as f:
            json.dump(snap, f, indent=1)
        print("snapshot of %d base tags, %d first_pair and %d pairs cases -> %s"
              % (len(bases), len(snap["first_pair"]), len(snap["pairs"]), BEFORE))
        return

    with open(BEFORE) as f:
        before = json.load(f)
    after = answers(before["bases"])
    changed = diff(before, after)
    checks = {"unchanged": {"ok": not changed, "changes": changed[:50],
                            "cases": {"tags": len(after["per_tag"]), "first_pair": len(after["first_pair"]),
                                      "pairs": len(after["pairs"])}},
              "new": check_new()}
    r = subprocess.run([sys.executable, os.path.join(paths.VARIANTS, "tests", "l0_skeleton.py")],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    checks["skeleton"] = {"ok": r.returncode == 0, "output": r.stdout.decode(errors="replace")[-800:]}

    report = {"checks": checks, "failed": [n for n, c in checks.items() if not c["ok"]]}
    report["pass"] = not report["failed"]
    path = os.path.join(GATE_DIR, "registry.json")
    with open(path + ".partial", "w") as f:
        json.dump(report, f, indent=2)
    os.replace(path + ".partial", path)
    for n, c in checks.items():
        print("  %-10s %s" % (n, "ok" if c["ok"] else "FAIL"))
    print("[gate registry] %s -> %s" % ("PASS" if report["pass"] else "FAIL", path))
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
