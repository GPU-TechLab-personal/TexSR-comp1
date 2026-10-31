#!/usr/bin/env python3
"""L6 (experiments), layer 0: every module of variants/experiments imports, and
nothing under it imports scripts/ or experiments/ (variants/README.md rule 1).

    python variants/tests/l6_experiments_imports.py      # in texsr/exp:torch2.0.0-cu117, CPU

Imports are made in a fresh interpreter per module so one module's sys.path
edits cannot hide another's missing import. Exit 0 = PASS.
"""
import os
import subprocess
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
ROOT = os.path.join(REPO, "variants", "experiments")


def modules():
    for d, _, files in os.walk(ROOT):
        for f in sorted(files):
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(d, f), REPO)[:-3].replace(os.sep, ".")
                yield rel[:-len(".__init__")] if rel.endswith(".__init__") else rel


PROBE = r"""
import sys, importlib
sys.path.insert(0, %r)
importlib.import_module(%r)
bad = sorted(m for m in sys.modules if m == "experiments" or m.startswith("experiments.")
             or m == "scripts" or m.startswith("scripts.") or m == "texsr_metrics")
print("BAD " + ",".join(bad) if bad else "OK")
"""


def main():
    fails = 0
    for mod in sorted(set(modules())):
        p = subprocess.run([sys.executable, "-c", PROBE % (REPO, mod)], capture_output=True, text=True,
                           cwd="/tmp" if os.path.isdir("/tmp") else None)
        out = p.stdout.strip().splitlines()[-1:] or [""]
        ok = p.returncode == 0 and out[0] == "OK"
        fails += not ok
        print("%s %s%s" % ("PASS" if ok else "FAIL", mod,
                           "" if ok else "  " + (out[0] if out[0] else p.stderr.strip().splitlines()[-1])))
    print("L6 imports: %s" % ("PASS" if not fails else "%d FAIL" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
