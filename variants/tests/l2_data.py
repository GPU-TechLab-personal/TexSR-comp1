#!/usr/bin/env python3
"""L2: the new data builders write the old data.

Host-side driver (plain Python; every step is a `docker run`, see l2_harness.py):

    python variants/tests/l2_data.py --stamp <stamp> [--main <main checkout>] \\
        [--data D:/experimentdata] [--only tdb,ms_rec,grouped,ms_pairs] [--ms-subset N]

Each old/new pair runs the old script (/workspace/scripts) and then the new one
(/workspace/variants) with the same arguments into the SAME output path - the old
output is moved aside in between - so manifests that record absolute paths can be
compared byte for byte too. Everything is written under
runs/variants_verify/deeptex_eval/<stamp>/l2/.

  tdb       textures_db: texsr_normalize -> texsr_make_pairs -> texsr_build_tfrecord
            -> texsr_verify_tfrecord, old chain vs new chain on the full released
            data/textures_db (299 sources). Folder trees byte-identical (every PNG,
            both manifests), the records identical record by record, the verify
            reports identical. Also: the new normalized/ and pair PNGs against the
            committed runs/deeptex/datasets/textures_db (informational - that is
            the same code on the same input, a month earlier).
  ms_rec    MatSynth records, recipes all / chratio / basecolor, from the committed
            pair folders runs/deeptex/datasets/matsynth: the old builder with the
            ps1's --hr/--lr/--repeat arguments vs matsynth/recipes.py. Identical
            record by record and in order; meta identical except sha256/bytes
            (protobuf map order); texsr_verify_tfrecord reports identical.
  grouped   the grouped record (render-loss arms), old builder vs new, --limit
            materials of the seed-0 order (0 = all 2904): records identical,
            order_sha256 identical, meta identical except sha256/bytes/created;
            texsr_verify_grouped old vs new reports identical except created/seconds.
  ms_pairs  matsynth_make_pairs: the new --check-eval must PASS (it rebuilds the 12
            evaluation materials from the test shard and requires pixel identity
            with runs/matsynth); and old vs new --limit-shards 1 into the same --out
            must give byte-identical pair PNGs, _meta/*.json and pairs_manifest.json.

Exit 0 only if every check passes. Results: l2/l2_report.json.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l2_harness as H  # noqa: E402

DT_SRC = "/workspace/external/DeepTextures/data/textures_db"
MS_PAIRS = "/workspace/runs/deeptex/datasets/matsynth"
RECIPES = {   # the ps1's $script:MatsynthRecipes (run_deeptex.ps1:539-543), for the OLD argv
    "all":       [("baseColor", 1), ("normal", 1), ("roughness", 1), ("metallic", 1), ("height", 1)],
    "chratio":   [("baseColor", 3), ("normal", 3), ("roughness", 1), ("metallic", 1), ("height", 1)],
    "basecolor": [("baseColor", 1)],
}
IGNORE_META = ("sha256", "bytes", "created")
IGNORE_REPORT = ("created", "seconds", "sha256", "bytes")


def strip(d, keys):
    if isinstance(d, dict):
        return {k: strip(v, keys) for k, v in d.items() if k not in keys}
    if isinstance(d, list):
        return [strip(v, keys) for v in d]
    return d


class L2:
    def __init__(self, env, rep):
        self.env, self.rep = env, rep
        self.c = env.out_c + "/l2"
        self.h = os.path.join(env.out, "l2")
        self.logs = os.path.join(self.h, "logs")
        os.makedirs(self.logs, exist_ok=True)

    def step(self, name, image, argv, ok_codes=(0,)):
        log = os.path.join(self.logs, name + ".log")
        code, sec = H.run(self.env, image, argv, log)
        if code not in ok_codes:
            self.rep.add(name, False, "exit %d (%s)" % (code, log), H.tail(log).splitlines(), sec)
            return None
        self.rep.clear(name)
        return sec

    def compare_records(self, name, a_c, b_c):
        """Run l2_records.py on two container paths; add a row."""
        js_c = "%s/%s.records.json" % (self.c, name)
        log = os.path.join(self.logs, name + "-records.log")
        code, sec = H.run(self.env, "deeptex",
                          ["python", "variants/tests/l2_records.py", a_c, b_c, "--json", js_c], log)
        js = self.env.to_host(js_c)
        if not os.path.exists(js):
            self.rep.add(name + ": records", False, "compare crashed (%s)" % log,
                         H.tail(log).splitlines(), sec)
            return None
        r = H.load_json(js)
        self.rep.add(name + ": records", r["same"],
                     "%d vs %d records, order_sha256 %s %s, file sha %s"
                     % (r["records_a"], r["records_b"], r["order_sha256_a"][:12],
                        "==" if r["order_sha256_a"] == r["order_sha256_b"] else "!= " + r["order_sha256_b"][:12],
                        "equal" if r["file_sha256_a"] == r["file_sha256_b"] else "differs (map order; not judged)"),
                     r["mismatches"], sec)
        return r

    def compare_json(self, name, a, b, ignore):
        if not (os.path.exists(a) and os.path.exists(b)):
            self.rep.add(name, False, "missing %s" % (a if not os.path.exists(a) else b))
            return
        d = H.numeric_diff(strip(H.load_json(a), ignore), strip(H.load_json(b), ignore))
        same = ("identical except %s" % ",".join(ignore)) if ignore else "identical"
        self.rep.add(name, not d, same if not d else "%d difference(s)" % len(d), d)

    # ------------------------------------------------------------ textures_db

    def tdb(self):
        ds_c, ds_h = self.c + "/tdb", os.path.join(self.h, "tdb")
        rec = "textures_db_lrx4.tfrecord"
        times = {}
        for side, root in (("old", "scripts/deeptex"), ("new", None)):
            H.remove(ds_h)
            norm = "%s/texsr_normalize.py" % (root or "variants/deeptex/paper")
            pairs = "%s/texsr_make_pairs.py" % (root or "variants/deeptex/paper")
            build = "%s/texsr_build_tfrecord.py" % (root or "variants/deeptex/upstream_runner")
            verify = "%s/texsr_verify_tfrecord.py" % (root or "variants/deeptex/upstream_runner")
            t = 0
            for name, image, argv in (
                    ("normalize", "dataprep", ["python", norm, "--in", DT_SRC, "--out", ds_c + "/normalized"]),
                    ("pairs", "deeptex", ["python", pairs, "--in", ds_c + "/normalized", "--out", ds_c,
                                          "--crop-size", "480", "--step", "240", "--lr-kernel", "bicubic",
                                          "--jobs", "8"]),
                    ("tfrecord", "deeptex", ["python", build, "--hr", ds_c + "/HR", "--lr", ds_c + "/LRx4",
                                             "--out", "%s/%s" % (ds_c, rec),
                                             "--pairs-manifest", ds_c + "/pairs_manifest.json",
                                             "--overwrite"]),
                    ("verify", "deeptex", ["python", verify, "--tfrecord", "%s/%s" % (ds_c, rec)])):
                s = self.step("tdb-%s-%s" % (side, name), image, argv)
                if s is None:
                    return
                t += s
            times[side] = t
            if side == "old":
                H.set_aside(ds_h)
        old_h = ds_h + ".old"
        # pairs_manifest.json as JSON, not bytes: its tiles_by_source is filled in
        # imap_unordered completion order (texsr_make_pairs.py:204-206), so two runs
        # of the OLD code already write its keys in different orders.
        c = H.compare_trees(old_h, ds_h, skip=(".tfrecord", "verify_report.json", "pairs_manifest.json"))
        self.compare_json("tdb: pairs_manifest.json (as JSON; key order is completion order)",
                          os.path.join(old_h, "pairs_manifest.json"),
                          os.path.join(ds_h, "pairs_manifest.json"), ())
        self.rep.add("tdb: normalized/, HR/, LRx4/, *_val/, normalize_manifest.json", H.tree_ok(c),
                     "%d files, %d identical; old chain %.0f s, new %.0f s"
                     % (c["n"], c["same"], times["old"], times["new"]),
                     c["differ"] + ["only old: %s" % x for x in c["only_a"]]
                     + ["only new: %s" % x for x in c["only_b"]])
        self.compare_records("tdb", self.c + "/tdb.old/" + rec, ds_c + "/" + rec)
        self.compare_json("tdb: tfrecord meta", os.path.join(old_h, rec + ".meta.json"),
                          os.path.join(ds_h, rec + ".meta.json"), IGNORE_META)
        self.compare_json("tdb: verify_report.json", os.path.join(old_h, "verify_report.json"),
                          os.path.join(ds_h, "verify_report.json"), IGNORE_REPORT)
        # Informational: the same code on the same input, as committed.
        committed = os.path.join(self.env.runs, "deeptex", "datasets", "textures_db")
        bad = []
        n = 0
        for sub in ("normalized", "HR", "LRx4", "HR_val", "LRx4_val"):
            cc = H.compare_trees(os.path.join(committed, sub), os.path.join(ds_h, sub))
            n += cc["n"]
            if not H.tree_ok(cc):
                bad.append("%s: %d differ, %d only committed, %d only new"
                           % (sub, cc["n_differ"], cc["n_only_a"], cc["n_only_b"]))
        self.rep.add("tdb: PNGs vs committed runs/deeptex/datasets/textures_db", not bad,
                     "%d files" % n, bad)
        H.remove(os.path.join(old_h, rec))
        H.remove(os.path.join(ds_h, rec))

    # ---------------------------------------------------------- MatSynth records

    def ms_rec(self, subset):
        src = MS_PAIRS
        if subset:
            # A subset root: the first N names of every map (sorted), copied.
            src = self.c + "/ms_subset"
            code = (
                "import os,shutil\n"
                "S=%r; D=%r; N=%d\n"
                "for m in ('baseColor','normal','roughness','metallic','height'):\n"
                "    names=sorted(f for f in os.listdir(os.path.join(S,m,'HR')) if f.endswith('.png'))[:N]\n"
                "    for s in ('HR','LRx4'):\n"
                "        os.makedirs(os.path.join(D,m,s),exist_ok=True)\n"
                "        for n in names: shutil.copyfile(os.path.join(S,m,s,n),os.path.join(D,m,s,n))\n"
                "shutil.copyfile(os.path.join(S,'pairs_manifest.json'),os.path.join(D,'pairs_manifest.json'))\n"
            ) % (MS_PAIRS, src, subset)
            if self.step("ms-subset", "dataprep", ["python", "-c", code]) is None:
                return
        for recipe, mix in RECIPES.items():
            out_c = "%s/ms_rec/%s" % (self.c, recipe)
            out_h = os.path.join(self.h, "ms_rec", recipe)
            rec = "matsynth_%s_lrx4.tfrecord" % recipe
            H.remove(out_h)
            H.remove(out_h + ".old")
            io = []
            for m, k in mix:
                io += ["--hr", "%s/%s/HR" % (src, m), "--lr", "%s/%s/LRx4" % (src, m)]
                if recipe != "all":
                    io += ["--repeat", str(k)]
            tail = ["--out", "%s/%s" % (out_c, rec), "--pairs-manifest", src + "/pairs_manifest.json",
                    "--overwrite"]
            ver = ["--tfrecord", "%s/%s" % (out_c, rec), "--align-sample", "16"]
            s_old = self.step("ms-%s-old-build" % recipe, "deeptex",
                              ["python", "scripts/deeptex/texsr_build_tfrecord.py"] + io + tail)
            if s_old is None:
                continue
            if self.step("ms-%s-old-verify" % recipe, "deeptex",
                         ["python", "scripts/deeptex/texsr_verify_tfrecord.py"] + ver) is None:
                continue
            H.set_aside(out_h)
            s_new = self.step("ms-%s-new-build" % recipe, "deeptex",
                              ["python", "variants/deeptex/matsynth/recipes.py", "--recipe", recipe,
                               "--root", src] + tail)
            if s_new is None:
                continue
            if self.step("ms-%s-new-verify" % recipe, "deeptex",
                         ["python", "variants/deeptex/upstream_runner/texsr_verify_tfrecord.py"] + ver) is None:
                continue
            print("  ms %s build: old %.0f s, new %.0f s" % (recipe, s_old, s_new))
            self.compare_records("ms %s" % recipe, "%s.old/%s" % (out_c, rec), "%s/%s" % (out_c, rec))
            self.compare_json("ms %s: tfrecord meta" % recipe, os.path.join(out_h + ".old", rec + ".meta.json"),
                              os.path.join(out_h, rec + ".meta.json"), IGNORE_META)
            self.compare_json("ms %s: verify_report.json" % recipe,
                              os.path.join(out_h + ".old", "verify_report.json"),
                              os.path.join(out_h, "verify_report.json"), IGNORE_REPORT)
            H.remove(out_h + ".old")
            H.remove(out_h)

    # ------------------------------------------------------------- grouped

    def grouped(self, limit):
        out_c = self.c + "/grouped"
        out_h = os.path.join(self.h, "grouped")
        rec = "matsynth_grouped_lrx4.tfrecord"
        H.remove(out_h)
        H.remove(out_h + ".old")
        argv = ["--root", MS_PAIRS, "--out", "%s/%s" % (out_c, rec), "--limit", str(limit), "--force"]
        ver = ["--tfrecord", "%s/%s" % (out_c, rec), "--root", MS_PAIRS]
        ok_verify = (0, 3) if limit else (0,)        # 3 = PARTIAL, a --limit build
        times = {}
        for side, b, v in (("old", "scripts/deeptex/texsr_build_grouped_tfrecord.py",
                            "scripts/deeptex/texsr_verify_grouped.py"),
                           ("new", "variants/deeptex/render_loss/build/texsr_build_grouped_tfrecord.py",
                            "variants/deeptex/render_loss/build/texsr_verify_grouped.py")):
            s = self.step("grouped-%s-build" % side, "deeptex", ["python", b] + argv)
            if s is None:
                return
            s2 = self.step("grouped-%s-verify" % side, "deeptex", ["python", v] + ver, ok_verify)
            if s2 is None:
                return
            times[side] = (s, s2)
            if side == "old":
                H.set_aside(out_h)
        print("  grouped build+verify: old %.0f+%.0f s, new %.0f+%.0f s" % (times["old"] + times["new"]))
        self.compare_records("grouped", "%s.old/%s" % (out_c, rec), "%s/%s" % (out_c, rec))
        mo = os.path.join(out_h + ".old", rec + ".meta.json")
        mn = os.path.join(out_h, rec + ".meta.json")
        self.compare_json("grouped: meta (order_sha256 included)", mo, mn, IGNORE_META)
        if os.path.exists(mo) and os.path.exists(mn):
            a, b = H.load_json(mo), H.load_json(mn)
            self.rep.add("grouped: order_sha256", a["order_sha256"] == b["order_sha256"],
                         "%s (%d records, --limit %d)" % (b["order_sha256"][:16], b["record_count"], limit))
            if not limit:
                committed = H.load_json(os.path.join(self.env.runs, "deeptex", "datasets",
                                                     "matsynth_grouped", rec + ".meta.json"))
                self.rep.add("grouped: order_sha256 vs committed meta",
                             committed["order_sha256"] == b["order_sha256"],
                             "committed %s" % committed["order_sha256"][:16])
        self.compare_json("grouped: verify_report.json", os.path.join(out_h + ".old", "verify_report.json"),
                          os.path.join(out_h, "verify_report.json"), IGNORE_REPORT)
        H.remove(out_h + ".old")
        H.remove(out_h)

    # ----------------------------------------------------- matsynth_make_pairs

    def ms_pairs(self, shards):
        if not self.env.data:
            self.rep.add("ms_pairs", False, "--data (the MatSynth parquet folder) not given")
            return
        for side, script in (("old", "scripts/deeptex/matsynth_make_pairs.py"),
                             ("new", "variants/deeptex/matsynth/matsynth_make_pairs.py")):
            log = os.path.join(self.logs, "ms_pairs-%s-check_eval.log" % side)
            code, sec = H.run(self.env, "dataprep", ["python", script, "--check-eval"], log)
            text = H.tail(log, 4)
            self.rep.add("ms_pairs: %s --check-eval" % side, code == 0 and "[PASS]" in text,
                         "exit %d, %.0f s" % (code, sec), text.splitlines())
        out_c, out_h = self.c + "/ms_pairs", os.path.join(self.h, "ms_pairs")
        H.remove(out_h)
        H.remove(out_h + ".old")
        times = {}
        for side, script in (("old", "scripts/deeptex/matsynth_make_pairs.py"),
                             ("new", "variants/deeptex/matsynth/matsynth_make_pairs.py")):
            s = self.step("ms_pairs-%s" % side, "dataprep",
                          ["python", script, "--data", "/data/MatSynth/data", "--out", out_c,
                           "--jobs", "1", "--limit-shards", str(shards)])
            if s is None:
                return
            times[side] = s
            if side == "old":
                H.set_aside(out_h)
        c = H.compare_trees(out_h + ".old", out_h)
        self.rep.add("ms_pairs: --limit-shards %d, same --out" % shards, H.tree_ok(c),
                     "%d files, %d identical; old %.0f s, new %.0f s"
                     % (c["n"], c["same"], times["old"], times["new"]),
                     c["differ"] + ["only old: %s" % x for x in c["only_a"]]
                     + ["only new: %s" % x for x in c["only_b"]])
        H.remove(out_h + ".old")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--main", default=None,
                    help="checkout whose runs/ and external/ to mount (default: this one)")
    ap.add_argument("--data", default=None, help="host folder mounted at /data (MatSynth)")
    ap.add_argument("--only", default="tdb,ms_rec,grouped,ms_pairs")
    ap.add_argument("--ms-subset", type=int, default=0,
                    help="ms_rec on the first N materials of every map (0 = all 2904)")
    ap.add_argument("--grouped-limit", type=int, default=0,
                    help="grouped record over the first N materials of the order (0 = all)")
    ap.add_argument("--shards", type=int, default=1, help="ms_pairs: --limit-shards")
    args = ap.parse_args()

    env = H.Env(args.main, args.data, args.stamp)
    os.makedirs(os.path.join(env.out, "l2"), exist_ok=True)
    rep = H.Report("L2", os.path.join(env.out, "l2", "l2_report.json"))
    t = L2(env, rep)
    only = set(args.only.split(","))
    if "tdb" in only:
        t.tdb()
    if "ms_rec" in only:
        t.ms_rec(args.ms_subset)
    if "grouped" in only:
        t.grouped(args.grouped_limit)
    if "ms_pairs" in only:
        t.ms_pairs(args.shards)
    rep.save()
    print("L2: %d check(s), %d FAIL -> %s" % (len(rep.rows), sum(not r["ok"] for r in rep.rows), rep.path))
    sys.exit(0 if rep.ok and rep.rows else 1)


if __name__ == "__main__":
    main()
