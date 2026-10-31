"""Host-side plumbing shared by the DeepTextures L2/L3 equivalence tests.

Runs on the host (plain Python 3.8+, no third-party imports). Every step of the
tests is a `docker run` of one of the project images with the same mount layout
docker/compose.yaml gives the services:

    <repo>                          -> /workspace                       (read-only)
    <external>                      -> /workspace/external              (read-only)
    <runs>                          -> /workspace/runs                  (read-only)
    <runs>/variants_verify          -> /workspace/runs/variants_verify  (the only rw mount)
    <data>                          -> /data                            (read-only, MatSynth)

<repo> is the checkout under test (a worktree works: its runs/ and external/ may be
empty, so --main points the other two mounts at the main checkout). The old code is
/workspace/scripts and the new code /workspace/variants of the SAME checkout, so
both sides always see identical inputs. Nothing outside runs/variants_verify/ can be
written: the repo, runs/ and the submodules are mounted read-only.

No container gets a GPU unless a step asks for one (gpu=True), and such a step first
waits until no other texsr* container is running (variants/README.md rule 8).
"""

import filecmp
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir))

IMAGES = {
    "deeptex": "texsr/deeptex:tf2.10.1",
    "matup": "texsr/matup:torch2.0.0-cu117",
    "exp": "texsr/exp:torch2.0.0-cu117",
    "dataprep": "texsr/dataprep:py3.11",
}
VERIFY_C = "/workspace/runs/variants_verify"


def host(p):
    return os.path.normpath(p).replace("\\", "/")


class Env:
    """Where things are on the host, and how a container path maps back."""

    def __init__(self, main_root=None, data=None, stamp=None, area="deeptex_eval"):
        main = main_root or REPO
        self.repo = REPO
        self.external = os.path.join(main, "external")
        self.runs = os.path.join(main, "runs")
        self.verify = os.path.join(self.runs, "variants_verify")
        self.data = data
        self.stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
        self.area = area
        self.out = os.path.join(self.verify, area, self.stamp)          # host
        self.out_c = "%s/%s/%s" % (VERIFY_C, area, self.stamp)          # container
        os.makedirs(self.out, exist_ok=True)

    def to_host(self, cpath):
        """/workspace/runs/variants_verify/... -> the host path."""
        assert cpath.startswith(VERIFY_C), cpath
        return os.path.normpath(os.path.join(self.verify, cpath[len(VERIFY_C):].lstrip("/")))

    def mounts(self):
        m = ["-v", "%s:/workspace:ro" % host(self.repo)]
        if os.path.normpath(self.external) != os.path.normpath(os.path.join(self.repo, "external")):
            m += ["-v", "%s:/workspace/external:ro" % host(self.external)]
        m += ["-v", "%s:/workspace/runs:ro" % host(self.runs),
              "-v", "%s:%s" % (host(self.verify), VERIFY_C)]
        if self.data:
            m += ["-v", "%s:/data:ro" % host(self.data)]
        return m


def other_texsr_containers():
    try:
        out = subprocess.check_output(["docker", "ps", "--format", "{{.Image}} {{.Names}}"],
                                      universal_newlines=True)
    except Exception:  # noqa: BLE001
        return []
    return [l for l in out.splitlines() if l.startswith("texsr/")]


def wait_for_gpu(poll=30, log=print):
    first = True
    while True:
        busy = other_texsr_containers()
        if not busy:
            return
        if first:
            log("  [gpu] waiting for: %s" % "; ".join(busy))
            first = False
        time.sleep(poll)


def run(env, image, argv, log_path, gpu=False, workdir="/workspace", extra_env=None,
        timeout=None):
    """docker run <image> <argv>, stdout+stderr -> log_path. Returns (code, seconds)."""
    if gpu:
        wait_for_gpu()
    cmd = ["docker", "run", "--rm", "--init"]
    if gpu:
        cmd += ["--gpus", "all"]
    cmd += ["-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "TF_CPP_MIN_LOG_LEVEL=2"]
    for k, v in (extra_env or {}).items():
        cmd += ["-e", "%s=%s" % (k, v)]
    cmd += env.mounts() + ["-w", workdir, IMAGES[image]] + list(argv)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("$ %s\n\n" % " ".join(cmd))
        fh.flush()
        e = dict(os.environ, MSYS_NO_PATHCONV="1")
        code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=e, timeout=timeout)
    return code, time.time() - t0


def tail(path, n=15):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tree_files(root):
    out = []
    for d, _, files in os.walk(root):
        for f in files:
            out.append(os.path.relpath(os.path.join(d, f), root).replace("\\", "/"))
    return sorted(out)


def compare_trees(a, b, skip=()):
    """Byte comparison of two folders. -> dict(n, only_a, only_b, differ)."""
    fa, fb = set(tree_files(a)), set(tree_files(b))
    fa = {f for f in fa if not any(s in f for s in skip)}
    fb = {f for f in fb if not any(s in f for s in skip)}
    differ = [f for f in sorted(fa & fb)
              if not filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False)]
    return {"n": len(fa | fb), "same": len(fa & fb) - len(differ),
            "only_a": sorted(fa - fb)[:20], "only_b": sorted(fb - fa)[:20],
            "n_only_a": len(fa - fb), "n_only_b": len(fb - fa),
            "differ": differ[:20], "n_differ": len(differ)}


def tree_ok(c):
    return c["n_only_a"] == 0 and c["n_only_b"] == 0 and c["n_differ"] == 0


def set_aside(path, suffix=".old"):
    """Move an old-code output out of the way so the new code writes to the SAME
    path (manifests record absolute paths)."""
    dst = path + suffix
    if os.path.exists(dst):
        shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
    os.replace(path, dst)
    return dst


def remove(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


def numeric_diff(a, b, path="", tol=0.0, out=None, limit=40):
    """Structural + numeric comparison of two JSON values. Floats compare exactly
    unless tol > 0; NaN equals NaN. -> list of difference strings."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append("%s/%s: only in %s" % (path, k, "new" if k not in a else "old"))
            else:
                numeric_diff(a[k], b[k], "%s/%s" % (path, k), tol, out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append("%s: length %d vs %d" % (path, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            numeric_diff(x, y, "%s[%d]" % (path, i), tol, out, limit)
    elif isinstance(a, bool) or isinstance(b, bool) or not (
            isinstance(a, (int, float)) and isinstance(b, (int, float))):
        if a != b:
            out.append("%s: %r vs %r" % (path, a, b))
    else:
        fa, fb = float(a), float(b)
        if fa != fa and fb != fb:
            return out
        if fa != fb and not (tol and abs(fa - fb) <= tol):
            out.append("%s: %r vs %r (diff %.3g)" % (path, a, b, fb - fa))
    return out


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def text_diff(a_path, b_path, limit=12):
    import difflib
    with open(a_path, encoding="utf-8", errors="replace") as f:
        a = f.read().splitlines()
    with open(b_path, encoding="utf-8", errors="replace") as f:
        b = f.read().splitlines()
    if a == b:
        return []
    d = list(difflib.unified_diff(a, b, "committed", "new", lineterm="", n=0))
    return d[:limit] + (["... %d more diff lines" % (len(d) - limit)] if len(d) > limit else [])


class Report:
    """Collects one layer's rows; written as JSON and as a markdown table."""

    def __init__(self, layer, path_json):
        """Rows of an earlier run with the same stamp are kept, so sections can be
        run one at a time (--only); a check run again replaces its row."""
        self.layer = layer
        self.path = path_json
        self.rows = []
        if os.path.exists(path_json):
            try:
                self.rows = load_json(path_json).get("rows", [])
            except ValueError:
                self.rows = []

    def add(self, name, ok, detail="", diff=None, seconds=None):
        self.rows = [r for r in self.rows if r["check"] != name]
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail,
                          "diff": diff or [], "seconds": seconds,
                          "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        print("%s  %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
        sys.stdout.flush()
        self.save()

    def clear(self, name):
        """A step that succeeded drops the FAIL row an earlier attempt left."""
        self.rows = [r for r in self.rows if r["check"] != name]

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"layer": self.layer, "rows": self.rows}, f, indent=1)

    @property
    def ok(self):
        return all(r["ok"] for r in self.rows)


# ------------------------------------------------------------------ report.md

def write_report(stamp_dir, l0_md=None):
    """<stamp>/report.md from l0.md, l2/l2_report.json, l3/l3_infer.json, l3/l3_reports.json."""
    parts = [("L0 static", None, l0_md or os.path.join(stamp_dir, "l0.md")),
             ("L2 data", os.path.join(stamp_dir, "l2", "l2_report.json"), None),
             ("L3 inference", os.path.join(stamp_dir, "l3", "l3_infer.json"), None),
             ("L3 reports", os.path.join(stamp_dir, "l3", "l3_reports.json"), None)]
    head = ["# DeepTextures eval equivalence - %s" % os.path.basename(stamp_dir.rstrip("/\\")), "",
            "| layer | checks | FAIL | stale committed file | verdict |", "|---|---|---|---|---|"]
    body, stale_rows = [], []
    for title, js, md in parts:
        if md:
            if not os.path.exists(md):
                head.append("| %s | - | - | - | not run |" % title)
                continue
            with open(md, encoding="utf-8") as f:
                text = f.read()
            n = text.count("\n| ") - 1
            n_fail = text.count("| FAIL |")
            head.append("| %s | %d | %d | - | %s |" % (title, n, n_fail, "PASS" if not n_fail else "FAIL"))
            body += ["", "## %s" % title, "", text.strip()]
            continue
        if not os.path.exists(js):
            head.append("| %s | - | - | - | not run |" % title)
            continue
        rows = load_json(js)["rows"]
        n_fail = sum(not r["ok"] for r in rows)
        n_stale = sum(bool(r.get("stale")) for r in rows)
        head.append("| %s | %d | %d | %d | %s |" % (title, len(rows), n_fail, n_stale,
                                                   "PASS" if rows and not n_fail else "FAIL"))
        body += ["", "## %s" % title, "", "| check | verdict | detail | s |", "|---|---|---|---|"]
        for r in rows:
            v = "PASS" if r["ok"] else "FAIL"
            if r.get("stale"):
                v = "PASS (stale)"
                stale_rows.append(r)
            sec = "" if r.get("seconds") is None else "%.0f" % r["seconds"]
            body.append("| %s | %s | %s | %s |" % (r["check"].replace("|", r"\|"), v,
                                                   str(r["detail"]).replace("|", r"\|"), sec))
        for r in rows:
            if not r["ok"] and r["diff"]:
                body += ["", "### FAIL: %s" % r["check"], "", "```"] + [str(x) for x in r["diff"][:30]] + ["```"]
    if stale_rows:
        body += ["", "## Differences that could not be matched", "",
                 "New == old on the same inputs, both != the committed file: the committed",
                 "file predates one of its inputs. First lines of each difference:", ""]
        for r in stale_rows:
            body += ["### %s" % r["check"], "", r["detail"], "", "```"] + [str(x) for x in r["diff"][:12]] + ["```", ""]
    out = os.path.join(stamp_dir, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(head + body) + "\n")
    return out


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "report":
        print(write_report(sys.argv[2]))
    else:
        sys.exit("usage: l2_harness.py report <runs/variants_verify/deeptex_eval/<stamp>>")
