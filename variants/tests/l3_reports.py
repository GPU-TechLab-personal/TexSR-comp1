#!/usr/bin/env python3
"""L3 (reports): the new report layer regenerates the committed reports.

Host-side driver (every step a `docker run`, see l2_harness.py):

    python variants/tests/l3_reports.py --stamp <stamp> [--main <main checkout>] \\
        [--only montages,base,arms,render,perceptual,drift,tiling] [--no-gpu]

Inputs are the frozen runs/deeptex/eval/*, runs/matsynth and runs/deeptex/* (all
mounted read-only); everything is written under
runs/variants_verify/deeptex_eval/<stamp>/l3/. Each section runs the NEW code and
compares with what is committed - JSON numerically (floats exactly, NaN == NaN),
markdown and CSV as text - and runs the OLD code on the same inputs so that a
mismatch against the committed file can be told apart from a refactoring error
(old == new, both != committed means the committed file is stale, not that the
refactor changed behaviour).

  montages    paper arm: deeptex_postprocess --deeptex-root runs/deeptex/eval/paper
              + deeptex_montage --collect, on a copy of runs/matsynth (they write
              into <mat>/compare/<stamp>/) -> runs/montages/<material>.json and
              summary.md
  base        the same without the DeepTextures column (compare/matup_postprocess +
              compare/matup_montage vs the old scripts without --deeptex-root):
              old vs new byte-identical
  arms        deeptex_arms_report for deeptex_matsynth, deeptex_data_ratio,
              deeptex_render_loss (arms read from the committed report.json) ->
              report.json, metrics_long.csv, summary.md, per_material.md
  render      svbrdf_render_eval (GPU, as committed) -> render_summary.md, and
              runs/deeptex/render_eval/main/{report.json,per_material.md,metrics_long.csv}
  perceptual  svbrdf_perceptual_eval (GPU, as committed) -> perceptual_report.json,
              perceptual_summary.md
  drift       deeptex_drift: old vs new on the CPU (identical text), and new on the
              GPU against the committed drift.md (GPU inference moves outputs by up
              to 1 LSB, DEEPTEX-DEVIATIONS 0.9, so this one is reported, with the
              largest cell difference, not required to be identical)
  tiling      deeptex_tiling_report -> runs/tiling/tiling_report.json (the script
              writes no summary.md; runs/tiling/summary.md is hand-written)

Exit 0 only if every required check passes. Results: l3/l3_reports.json.
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l2_harness as H  # noqa: E402

EVAL_ROOT = "/workspace/runs/matsynth"
MATERIALS = ("0559_brick_uneven_stones", "0844_syntetic_foam", "1587_cobblestones",
             "2119_corten_steel", "acg_concrete_026", "acg_metal_026", "acg_plastic_009",
             "acg_roofing_tiles_006", "acg_wood_floor_060", "tc_metal_029",
             "th_brick_wall_005", "th_brown_mud_leaves_001")
REPORTS = ("deeptex_matsynth", "deeptex_data_ratio", "deeptex_render_loss")


class L3:
    def __init__(self, env, rep, gpu):
        self.env, self.rep, self.gpu = env, rep, gpu
        self.c = env.out_c + "/l3"
        self.h = os.path.join(env.out, "l3")
        self.logs = os.path.join(self.h, "logs")
        os.makedirs(self.logs, exist_ok=True)
        self.runs = env.runs

    def step(self, name, image, argv, gpu=False, ok_codes=(0,)):
        log = os.path.join(self.logs, name + ".log")
        code, sec = H.run(self.env, image, argv, log, gpu=gpu)
        if code not in ok_codes:
            self.rep.add(name, False, "exit %d (%s)" % (code, log), H.tail(log).splitlines(), sec)
            return None
        self.rep.clear(name)
        return sec

    def _stale(self, name, n, biggest, old_same, required, d):
        """new != committed. If the OLD code, on the same inputs, writes exactly what
        the new code writes, the refactor is not the cause: the committed file
        predates one of its inputs. That is recorded (stale=True, with the size of
        the difference), not failed. Otherwise it fails."""
        if old_same:
            self.rep.add(name, True, "STALE committed file: %d value(s) differ, largest |diff| %s; "
                         "the old code writes the same as the new" % (n, biggest), d)
            self.rep.rows[-1]["stale"] = True
        else:
            self.rep.add(name, not required, "%d difference(s)%s" % (
                n, "" if required else " (informational)"), d)

    def json_vs(self, name, committed, new, ignore=(), required=True, old=None):
        if not os.path.exists(new):
            self.rep.add(name, False, "not written: %s" % new)
            return False
        a, b = H.load_json(committed), H.load_json(new)
        for k in ignore:
            a.pop(k, None)
            b.pop(k, None)
        d = H.numeric_diff(a, b, limit=100000)
        if not d:
            self.rep.add(name, True, "numerically identical")
            return True
        biggest = max([abs(float(m.group(1))) for m in
                       (re.search(r"\(diff (-?[0-9.e+-]+|nan|inf)\)", x) for x in d) if m] or [0.0])
        old_same = bool(old and os.path.exists(old)
                        and not H.numeric_diff(H.load_json(old), H.load_json(new), limit=1))
        self._stale(name, len(d), "%.3g" % biggest, old_same, required, d[:12])
        return False

    def text_vs(self, name, committed, new, required=True, old=None):
        if not os.path.exists(new):
            self.rep.add(name, False, "not written: %s" % new)
            return False
        d = H.text_diff(committed, new, limit=100000)
        if not d:
            self.rep.add(name, True, "text identical")
            return True
        changed = sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
        old_same = bool(old and os.path.exists(old) and not H.text_diff(old, new))
        self._stale(name, changed, "n/a (text, %d line(s))" % changed, old_same, required, d[:12])
        return False

    def tree_vs(self, name, a, b, skip=()):
        c = H.compare_trees(a, b, skip)
        self.rep.add(name, H.tree_ok(c), "%d files, %d byte-identical" % (c["n"], c["same"]),
                     c["differ"] + ["only old: %s" % x for x in c["only_a"]]
                     + ["only new: %s" % x for x in c["only_b"]])

    def copy_matsynth(self, dst_c):
        """runs/matsynth -> dst: SVBRDF, GT, meta.json and the newest MatUp run only
        (the one every script picks; sorted(...)[-1])."""
        code = (
            "import os,shutil,glob\n"
            "S=%r; D=%r\n"
            "for m in sorted(os.listdir(S)):\n"
            "    s=os.path.join(S,m); d=os.path.join(D,m)\n"
            "    if not os.path.isdir(os.path.join(s,'SVBRDF')): continue\n"
            "    for sub in ('SVBRDF','GT'): shutil.copytree(os.path.join(s,sub),os.path.join(d,sub))\n"
            "    shutil.copyfile(os.path.join(s,'meta.json'),os.path.join(d,'meta.json'))\n"
            "    runs=sorted(x for x in glob.glob(os.path.join(s,'matup-output-%%s'%%m,'*')) if os.path.isdir(x))\n"
            "    shutil.copytree(runs[-1],os.path.join(d,'matup-output-%%s'%%m,os.path.basename(runs[-1])))\n"
        ) % (EVAL_ROOT, dst_c)
        return self.step("copy-matsynth-" + os.path.basename(os.path.dirname(dst_c)), "dataprep",
                         ["python", "-c", code])

    # ------------------------------------------------------------- montages

    def montages(self):
        roots = {}
        for side in ("old", "new"):
            root_c = "%s/montages/%s/matsynth" % (self.c, side)
            H.remove(self.env.to_host(root_c))
            if self.copy_matsynth(root_c) is None:
                return
            roots[side] = root_c
        dtx = "/workspace/runs/deeptex/eval/paper"
        steps = {
            "old": (["python", "scripts/compare/matup_postprocess.py"],
                    ["python", "scripts/compare/matup_montage.py"]),
            "new": (["python", "variants/deeptex/paper/deeptex_postprocess.py"],
                    ["python", "variants/deeptex/paper/deeptex_montage.py"]),
        }
        for side, (pp, mt) in steps.items():
            col = "%s/montages/%s/collect" % (self.c, side)
            if self.step("montages-%s-postprocess" % side, "dataprep",
                         pp + ["--root", roots[side], "--all", "--metrics", "--deeptex-root", dtx]) is None:
                return
            if self.step("montages-%s-montage" % side, "dataprep",
                         mt + ["--root", roots[side], "--all", "--collect", col]) is None:
                return
        new_col = os.path.join(self.h, "montages", "new", "collect")
        old_col = os.path.join(self.h, "montages", "old", "collect")
        committed = os.path.join(self.runs, "montages")
        for m in MATERIALS:
            self.json_vs("montages: runs/montages/%s.json" % m, os.path.join(committed, m + ".json"),
                         os.path.join(new_col, m + ".json"), old=os.path.join(old_col, m + ".json"))
        self.text_vs("montages: runs/montages/summary.md", os.path.join(committed, "summary.md"),
                     os.path.join(new_col, "summary.md"), old=os.path.join(old_col, "summary.md"))
        self.tree_vs("montages: old vs new collect/ (PNG + JSON + md)", old_col, new_col)
        self.tree_vs("montages: old vs new <mat>/compare/ (every column, sheet, metrics.json)",
                     os.path.join(self.h, "montages", "old", "matsynth"),
                     os.path.join(self.h, "montages", "new", "matsynth"))
        n_png = 0
        bad = []
        for m in MATERIALS:
            for suffix in (".png", "_deeptex.png"):
                a, b = os.path.join(committed, m + suffix), os.path.join(new_col, m + suffix)
                n_png += 1
                if not (os.path.exists(b) and H.sha256_file(a) == H.sha256_file(b)):
                    bad.append(m + suffix)
        self.rep.add("montages: sheets vs committed runs/montages/*.png (informational)", True,
                     "%d/%d byte-identical" % (n_png - len(bad), n_png), bad)
        for side in ("old", "new"):
            H.remove(os.path.join(self.h, "montages", side, "matsynth"))

    def base(self):
        roots = {}
        for side in ("old", "new"):
            root_c = "%s/base/%s/matsynth" % (self.c, side)
            H.remove(self.env.to_host(root_c))
            if self.copy_matsynth(root_c) is None:
                return
            roots[side] = root_c
        steps = {
            "old": (["python", "scripts/compare/matup_postprocess.py"],
                    ["python", "scripts/compare/matup_montage.py"]),
            "new": (["python", "variants/compare/matup_postprocess.py"],
                    ["python", "variants/compare/matup_montage.py"]),
        }
        for side, (pp, mt) in steps.items():
            col = "%s/base/%s/collect" % (self.c, side)
            if self.step("base-%s-postprocess" % side, "dataprep",
                         pp + ["--root", roots[side], "--all", "--metrics"]) is None:
                return
            if self.step("base-%s-montage" % side, "dataprep",
                         mt + ["--root", roots[side], "--all", "--collect", col]) is None:
                return
        self.tree_vs("base (no DeepTextures column): old vs new collect/",
                     os.path.join(self.h, "base", "old", "collect"),
                     os.path.join(self.h, "base", "new", "collect"))
        self.tree_vs("base: old vs new <mat>/compare/",
                     os.path.join(self.h, "base", "old", "matsynth"),
                     os.path.join(self.h, "base", "new", "matsynth"))
        for side in ("old", "new"):
            H.remove(os.path.join(self.h, "base", side, "matsynth"))

    # ---------------------------------------------------------------- arms

    def arms(self):
        for name in REPORTS:
            committed = os.path.join(self.runs, "montages", name)
            arms = H.load_json(os.path.join(committed, "report.json"))["arms"]
            argv = []
            for tag, d in arms.items():
                argv += ["--arm", "%s=%s" % (tag, d)]
            outs = {}
            for side, script in (("old", "scripts/deeptex/deeptex_arms_report.py"),
                                 ("new", "variants/deeptex/matsynth/deeptex_arms_report.py")):
                out_c = "%s/arms/%s/%s" % (self.c, side, name)
                H.remove(self.env.to_host(out_c))
                if self.step("arms-%s-%s" % (name, side), "dataprep",
                             ["python", script] + argv + ["--out", out_c]) is None:
                    break
                outs[side] = self.env.to_host(out_c)
            if len(outs) < 2:
                continue
            new = outs["new"]
            self.json_vs("%s: report.json" % name, os.path.join(committed, "report.json"),
                         os.path.join(new, "report.json"), old=os.path.join(outs["old"], "report.json"))
            for f in ("metrics_long.csv", "summary.md", "per_material.md"):
                self.text_vs("%s: %s" % (name, f), os.path.join(committed, f), os.path.join(new, f),
                             old=os.path.join(outs["old"], f))
            self.tree_vs("%s: old vs new (montages, overviews, sheets, tables)" % name,
                         outs["old"], outs["new"])
            # The full-size sheets are hundreds of MB; the comparison is done.
            for side in outs:
                H.remove(os.path.join(outs[side], "sheets"))

    # -------------------------------------------------------------- render

    def _dt_arms(self, report):
        argv = []
        for c in report["columns"]:
            if c["kind"] == "dt":
                argv += ["--arm", "%s=%s" % (c["tag"], c["dir"])]
        return argv

    def render(self):
        ref = os.path.join(self.runs, "deeptex", "render_eval", "main")
        argv = ["--root", EVAL_ROOT] + self._dt_arms(H.load_json(os.path.join(ref, "report.json")))
        outs = {}
        for side, script in (("new", "variants/deeptex/render_loss/eval/svbrdf_render_eval.py"),
                             ("old", "scripts/compare/svbrdf_render_eval.py")):
            out_c = "%s/render/%s" % (self.c, side)
            H.remove(self.env.to_host(out_c))
            if self.step("render-%s" % side, "matup", ["python", script] + argv + ["--out", out_c],
                         gpu=self.gpu) is None:
                return
            outs[side] = self.env.to_host(out_c)
        new = outs["new"]
        committed_md = os.path.join(self.runs, "montages", "deeptex_render_loss", "render_summary.md")
        old = outs["old"]
        self.text_vs("render: render_summary.md", committed_md, os.path.join(new, "summary.md"),
                     old=os.path.join(old, "summary.md"))
        self.json_vs("render: render_eval/main/report.json", os.path.join(ref, "report.json"),
                     os.path.join(new, "report.json"), old=os.path.join(old, "report.json"))
        for f in ("per_material.md", "metrics_long.csv"):
            self.text_vs("render: render_eval/main/%s" % f, os.path.join(ref, f), os.path.join(new, f),
                         old=os.path.join(old, f))
        self.tree_vs("render: old vs new (sheets, overview, tables)", outs["old"], outs["new"])

    def perceptual(self):
        committed = os.path.join(self.runs, "montages", "deeptex_render_loss")
        rep = H.load_json(os.path.join(committed, "perceptual_report.json"))
        argv = ["--root", EVAL_ROOT] + self._dt_arms(rep)
        argv += ["--device", rep["constants"]["device"]]
        outs = {}
        for side, script in (("new", "variants/deeptex/render_loss/eval/svbrdf_perceptual_eval.py"),
                             ("old", "scripts/compare/svbrdf_perceptual_eval.py")):
            out_c = "%s/perceptual/%s" % (self.c, side)
            H.remove(self.env.to_host(out_c))
            if self.step("perceptual-%s" % side, "exp", ["python", script] + argv + ["--out", out_c],
                         gpu=self.gpu) is None:
                return
            outs[side] = self.env.to_host(out_c)
        new = outs["new"]
        self.json_vs("perceptual: perceptual_report.json", os.path.join(committed, "perceptual_report.json"),
                     os.path.join(new, "perceptual_report.json"),
                     old=os.path.join(outs["old"], "perceptual_report.json"))
        self.text_vs("perceptual: perceptual_summary.md", os.path.join(committed, "perceptual_summary.md"),
                     os.path.join(new, "perceptual_summary.md"),
                     old=os.path.join(outs["old"], "perceptual_summary.md"))
        self.tree_vs("perceptual: old vs new", outs["old"], outs["new"])

    # --------------------------------------------------------------- drift

    def drift(self, cpu_n, gpu_tags):
        """CPU old vs new on matsynth_all (3 checkpoints, --n cpu_n); GPU new vs the
        committed drift.md for gpu_tags (runs/deeptex/drift/<tag>)."""
        src = os.path.join(self.runs, "deeptex", "drift", "matsynth_all")
        dj = H.load_json(os.path.join(src, "drift.json"))
        base = ["--cfg", dj["cfg"], "--ckpt-dir", dj["ckpt_dir"], "--pairs", dj["pairs"],
                "--eval-root", dj["eval_root"], "--eval-map", dj["eval_map"], "--seed", str(dj["seed"])]
        outs = {}
        for side, script in (("old", "scripts/deeptex/deeptex_drift.py"),
                             ("new", "variants/deeptex/matsynth/deeptex_drift.py")):
            # Same --out for both (drift.md prints the arguments), the old one set aside.
            out_c = "%s/drift/cpu/matsynth_all" % self.c
            if side == "new":
                H.set_aside(self.env.to_host(out_c))
            else:
                H.remove(self.env.to_host(out_c))
                H.remove(self.env.to_host(out_c) + ".old")
            argv = ["python", script] + base + ["--n", str(cpu_n), "--sheet-material", "0844_syntetic_foam",
                                                "--out", out_c]
            if self.step("drift-cpu-%s" % side, "deeptex", argv) is None:
                return
            outs[side] = self.env.to_host(out_c)
        self.tree_vs("drift (CPU, matsynth_all, --n %d): old vs new drift.md/json, sheets, outputs" % cpu_n,
                     outs["new"] + ".old", outs["new"])
        if not self.gpu:
            return
        for tag in gpu_tags:
            src = os.path.join(self.runs, "deeptex", "drift", tag)
            dj = H.load_json(os.path.join(src, "drift.json"))
            out_c = "%s/drift/gpu/%s" % (self.c, tag)
            H.remove(self.env.to_host(out_c))
            # The committed --out, so drift.md's header line matches; -> a copy here.
            argv = ["python", "variants/deeptex/matsynth/deeptex_drift.py", "--cfg", dj["cfg"],
                    "--ckpt-dir", dj["ckpt_dir"], "--pairs", dj["pairs"], "--n", str(dj["n"]),
                    "--seed", str(dj["seed"]), "--eval-root", dj["eval_root"],
                    "--eval-map", dj["eval_map"], "--out", out_c]
            if self.step("drift-gpu-%s" % tag, "deeptex", argv, gpu=True) is None:
                continue
            new = self.env.to_host(out_c)
            d = H.numeric_diff(H.load_json(os.path.join(src, "drift.json")),
                               H.load_json(os.path.join(new, "drift.json")), limit=400)
            biggest = 0.0
            for line in d:
                m = re.search(r"diff (-?[\d.e+-]+)\)", line)
                if m:
                    biggest = max(biggest, abs(float(m.group(1))))
            self.text_vs("drift %s (GPU): drift.md vs committed" % tag, os.path.join(src, "drift.md"),
                         os.path.join(new, "drift.md"), required=False)
            self.rep.add("drift %s (GPU): drift.json vs committed (informational)" % tag, True,
                         "%d value(s) differ, largest |diff| %.3g (8-bit levels)" % (len(d), biggest), d[:6])

    # -------------------------------------------------------------- tiling

    def tiling(self):
        outs = {}
        for side, script in (("old", "scripts/deeptex/deeptex_tiling_report.py"),
                             ("new", "variants/deeptex/tiling/deeptex_tiling_report.py")):
            out_c = "%s/tiling/%s" % (self.c, side)
            H.remove(self.env.to_host(out_c))
            # The script never creates --out, and cv2.imwrite fails silently into
            # a missing folder (only the final json open raises) - see tiling/README.md.
            os.makedirs(self.env.to_host(out_c))
            if self.step("tiling-%s" % side, "dataprep", ["python", script, "--out", out_c]) is None:
                return
            outs[side] = self.env.to_host(out_c)
        committed = os.path.join(self.runs, "tiling")
        self.json_vs("tiling: runs/tiling/tiling_report.json", os.path.join(committed, "tiling_report.json"),
                     os.path.join(outs["new"], "tiling_report.json"),
                     old=os.path.join(outs["old"], "tiling_report.json"))
        self.tree_vs("tiling: old vs new", outs["old"], outs["new"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--main", default=None)
    ap.add_argument("--only", default="montages,base,arms,render,perceptual,drift,tiling")
    ap.add_argument("--no-gpu", action="store_true",
                    help="render/perceptual on the CPU (then they cannot match the committed "
                         "GPU numbers exactly); skips the GPU drift reference")
    ap.add_argument("--drift-n", type=int, default=6, help="drift CPU check: pairs per checkpoint")
    ap.add_argument("--drift-gpu", default="matsynth_all,render_r1_baseColor",
                    help="runs/deeptex/drift/<tag> regenerated on the GPU for reference")
    args = ap.parse_args()

    env = H.Env(args.main, None, args.stamp)
    os.makedirs(os.path.join(env.out, "l3"), exist_ok=True)
    rep = H.Report("L3 reports", os.path.join(env.out, "l3", "l3_reports.json"))
    t = L3(env, rep, not args.no_gpu)
    only = args.only.split(",")
    for sec in only:
        if sec == "drift":
            t.drift(args.drift_n, [x for x in args.drift_gpu.split(",") if x])
        else:
            getattr(t, sec)()
    rep.save()
    print("L3 reports: %d check(s), %d FAIL -> %s"
          % (len(rep.rows), sum(not r["ok"] for r in rep.rows), rep.path))
    sys.exit(0 if rep.ok and rep.rows else 1)


if __name__ == "__main__":
    main()
