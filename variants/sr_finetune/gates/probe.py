"""Run one check in a child process and get its result back as JSON.

The official code bases cannot share a process (paths.use_kair() and
use_swinir_official() both own the top-level `models` / `utils`), so a gate that
looks at more than one starts itself again as `<script> --probe <which> ...`, the
child prints one PROBE line, and run() returns it. A child that crashes before
printing comes back as {"ok": False, "exit": ..., "output": <its last 4000 chars>}.
"""
import json
import subprocess
import sys

TAG = "PROBE "


def run(script, which, *args):
    r = subprocess.run([sys.executable, script, "--probe", which] + list(args),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = r.stdout.decode(errors="replace")
    lines = [line for line in text.splitlines() if line.startswith(TAG)]
    if lines:
        return json.loads(lines[-1][len(TAG):])
    return {"ok": False, "exit": r.returncode, "output": text[-4000:]}


def emit(result):
    print(TAG + json.dumps(result))
