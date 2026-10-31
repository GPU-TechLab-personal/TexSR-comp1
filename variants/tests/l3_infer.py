#!/usr/bin/env python3
"""L3 (inference): the new deeptex_compare / eval_arms write the old bytes.

Host-side driver (every step a `docker run`, see l2_harness.py):

    python variants/tests/l3_infer.py --stamp <stamp> [--main <main checkout>] [--no-gpu]
        [--materials 0844_syntetic_foam,2119_corten_steel,acg_plastic_009]

  cpu paper     old scripts/deeptex/deeptex_compare.py (default maps baseColor,render)
                vs variants/deeptex/infer/deeptex_compare.py, the textures_db
                checkpoint, all 12 materials, on the CPU, into the same --out:
                every PNG and sr_manifest.json byte-identical. REQUIRED.
  cpu all-maps  old deeptex_compare.py --maps all --scalar-collapse mean vs
                variants/deeptex/matsynth/eval_arms.py, the matsynth_all
                checkpoint, CPU, same
                --out, all 12 materials in one process: byte-identical. And
                --scalar-collapse paper on the first of --materials. REQUIRED.
  gpu           (reference, not required) the new code on the GPU against the
                committed runs/deeptex/eval/paper and .../matsynth_all, which were
                written on the GPU: largest pixel difference must be <= 1 LSB (GPU
                inference is not bit-reproducible, DEEPTEX-DEVIATIONS 0.9).

Results: l3/l3_infer.json.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l2_harness as H  # noqa: E402

ROOT = "/workspace/runs/matsynth"
PAPER = ("/workspace/runs/deeptex/paper/checkpoints/esrgan_40k",
         "/workspace/configs/deeptex/esrgan_paper.yaml",
         "/workspace/variants/deeptex/paper/esrgan_paper.yaml")
MSALL = ("/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k",
         "/workspace/configs/deeptex/esrgan_matsynth.yaml",
         "/workspace/variants/deeptex/matsynth/esrgan_matsynth.yaml")


class L3I:
    def __init__(self, env, rep):
        self.env, self.rep = env, rep
        self.c = env.out_c + "/l3/infer"
        self.h = os.path.join(env.out, "l3", "infer")
        self.logs = os.path.join(env.out, "l3", "logs")
        os.makedirs(self.logs, exist_ok=True)

    def step(self, name, image, argv, gpu=False):
        log = os.path.join(self.logs, name + ".log")
        code, sec = H.run(self.env, image, argv, log, gpu=gpu)
        if code != 0:
            self.rep.add(name, False, "exit %d (%s)" % (code, log), H.tail(log).splitlines(), sec)
            return None
        self.rep.clear(name)
        return sec

    def old_new(self, label, runs_old, runs_new, out_c):
        """Each runs_* is a list of (name, argv); old into out_c, set aside, new into out_c."""
        out_h = self.env.to_host(out_c)
        H.remove(out_h)
        H.remove(out_h + ".old")
        t = {}
        for side, runs in (("old", runs_old), ("new", runs_new)):
            t[side] = 0.0
            for name, argv in runs:
                s = self.step("%s-%s" % (name, side), "deeptex", argv)
                if s is None:
                    return
                t[side] += s
            if side == "old":
                H.set_aside(out_h)
        c = H.compare_trees(out_h + ".old", out_h)
        self.rep.add(label, H.tree_ok(c),
                     "%d files, %d byte-identical; old %.0f s, new %.0f s"
                     % (c["n"], c["same"], t["old"], t["new"]),
                     c["differ"] + ["only old: %s" % x for x in c["only_a"]]
                     + ["only new: %s" % x for x in c["only_b"]])

    def cpu_paper(self):
        ck, cfg_old, cfg_new = PAPER
        out_c = self.c + "/cpu_paper"
        common = ["--ckpt-dir", ck, "--root", ROOT, "--out", out_c]
        self.old_new("infer cpu: deeptex_compare (baseColor,render), textures_db ckpt, 12 materials",
                     [("infer-cpu-paper", ["python", "scripts/deeptex/deeptex_compare.py", "--cfg", cfg_old] + common)],
                     [("infer-cpu-paper", ["python", "variants/deeptex/infer/deeptex_compare.py", "--cfg", cfg_new] + common)],
                     out_c)

    def cpu_all(self, materials):
        ck, cfg_old, cfg_new = MSALL
        # mean: every material in one process (no --material), as eval-arms runs it;
        # paper: the other collapse on the first of --materials.
        for collapse, mats in (("mean", [""]), ("paper", materials[:1])):
            out_c = "%s/cpu_all_%s" % (self.c, collapse)
            old, new = [], []
            for m in mats:
                common = ["--ckpt-dir", ck, "--root", ROOT, "--out", out_c,
                          "--scalar-collapse", collapse] + (["--material", m] if m else [])
                old.append(("infer-cpu-all-%s-%s" % (collapse, m or "all"),
                            ["python", "scripts/deeptex/deeptex_compare.py", "--cfg", cfg_old,
                             "--maps", "all"] + common))
                new.append(("infer-cpu-all-%s-%s" % (collapse, m or "all"),
                            ["python", "variants/deeptex/matsynth/eval_arms.py", "--cfg", cfg_new] + common))
            self.old_new("infer cpu: --maps all --scalar-collapse %s vs eval_arms, matsynth_all ckpt, %s"
                         % (collapse, ",".join(mats) if mats[0] else "12 materials"), old, new, out_c)

    def gpu(self):
        for label, (ck, _, cfg), script, extra, ref in (
                ("paper", PAPER, "variants/deeptex/infer/deeptex_compare.py", [],
                 "/workspace/runs/deeptex/eval/paper"),
                ("matsynth_all", MSALL, "variants/deeptex/matsynth/eval_arms.py",
                 ["--scalar-collapse", "mean"], "/workspace/runs/deeptex/eval/matsynth_all")):
            out_c = "%s/gpu_%s" % (self.c, label)
            H.remove(self.env.to_host(out_c))
            if self.step("infer-gpu-%s" % label, "deeptex",
                         ["python", script, "--cfg", cfg, "--ckpt-dir", ck, "--root", ROOT,
                          "--out", out_c] + extra, gpu=True) is None:
                continue
            js_c = "%s/gpu_%s.pngdiff.json" % (self.c, label)
            if self.step("infer-gpu-%s-diff" % label, "dataprep",
                         ["python", "variants/tests/l3_pngdiff.py", ref, out_c, "--json", js_c]) is None:
                continue
            r = H.load_json(self.env.to_host(js_c))
            self.rep.add("infer gpu (reference): %s vs committed %s" % (label, ref.rsplit("/", 1)[-1]),
                         r["max_abs_diff"] <= 1 and not r["problems"],
                         "%d PNGs, %d pixel-identical, max |diff| %d LSB"
                         % (r["files"], r["pixel_identical_files"], r["max_abs_diff"]),
                         r["problems"] + ["%s max %d (%d px)" % (w["file"], w["max_abs"], w["n_diff"])
                                          for w in r["worst"][:5] if w["max_abs"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--main", default=None)
    ap.add_argument("--only", default="cpu_paper,cpu_all,gpu")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--materials", default="0844_syntetic_foam,2119_corten_steel,acg_plastic_009")
    args = ap.parse_args()

    env = H.Env(args.main, None, args.stamp)
    os.makedirs(os.path.join(env.out, "l3"), exist_ok=True)
    rep = H.Report("L3 inference", os.path.join(env.out, "l3", "l3_infer.json"))
    t = L3I(env, rep)
    only = args.only.split(",")
    if "cpu_paper" in only:
        t.cpu_paper()
    if "cpu_all" in only:
        t.cpu_all(args.materials.split(","))
    if "gpu" in only and not args.no_gpu:
        t.gpu()
    rep.save()
    print("L3 inference: %d check(s), %d FAIL -> %s"
          % (len(rep.rows), sum(not r["ok"] for r in rep.rows), rep.path))
    sys.exit(0 if rep.ok and rep.rows else 1)


if __name__ == "__main__":
    main()
