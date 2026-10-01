#!/usr/bin/env python3
"""Score several DeepTextures arms against MatUp and GT on every map and the render.

Run it in the dataprep container, after deeptex_compare.py --maps all has written
each arm's outputs:

    docker compose run --rm dataprep python scripts/deeptex/deeptex_arms_report.py \\
        --arm textures_db=/workspace/runs/deeptex/eval/paper_allmaps \\
        --arm matsynth_all=/workspace/runs/deeptex/eval/matsynth_all \\
        --out /workspace/runs/deeptex/report_matsynth_all

Written under --out (and nowhere else):

    sheets/<material>_<map>.png    LR / bilinear / MatUp / one column per arm / GT
    overview_<map>.png             the 12 material sheets stacked, half size
    report.json                    every number, per material
    summary.md                     means over the materials, one table per map

WHY A SEPARATE SCRIPT
---------------------
matup_postprocess.py --deeptex-root already adds ONE DeepTextures column, on
baseColor and render. It writes into runs/matsynth/<mat>/compare/<stamp>/, which
matup_montage.py --collect turns into the committed runs/montages/. A second arm
or four more maps there would rewrite committed artefacts. So this script imports
the same column-building code (identical crops, border and DirectX->OpenGL flip,
hence numbers comparable with the committed ones) and writes only to --out.

METRICS
-------
* PSNR / SSIM at 8 bits (texsr_metrics.full_reference via matup_postprocess):
  DeepTextures emits 8-bit, so MatUp's 16 bits are rounded away first.
* |grad| (matup_postprocess.sharpness), reference-free, as in the committed JSON.
* normal: mean angular error in degrees, also on 8-bit-rounded inputs. PSNR on a
  normal map scores pixels, not directions.
* channel_spread from each arm's sr_manifest.json: how far apart the network's
  three output channels were on a 1-channel map. Trained on replicated grey maps,
  the MatSynth arm should push this toward 0; the textures_db arm never saw them.

A metallic GT is a constant 0 on 9 of the 12 materials; a column that reproduces
it exactly has infinite PSNR. Means are taken over finite values and the number
of exact reproductions is reported beside them.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "compare"))
import matup_postprocess as post  # noqa: E402

BASE_ORDER = ["lr", "bilinear", "matup"]


def arm_col(label):
    return "dt_" + label


def angular_error_deg(pred, gt):
    """Mean angle between decoded normals, both rounded to 8 bits first."""
    def vec(a):
        v = np.rint(post.to_float(a) * 255.0) / 255.0 * 2.0 - 1.0
        n = np.linalg.norm(v, axis=2, keepdims=True)
        return v / np.maximum(n, 1e-8)
    d = np.clip((vec(pred) * vec(gt)).sum(axis=2), -1.0, 1.0)
    return float(np.degrees(np.arccos(d)).mean())


def newest_run(mat_dir, name, want):
    runs = sorted(d for d in glob.glob(os.path.join(mat_dir, "matup-output-%s" % name, "*"))
                  if os.path.isdir(d))
    if want:
        runs = [r for r in runs if os.path.basename(r) == want]
    return runs[-1] if runs else None


def channel_spread(arm_dir, name, m):
    p = os.path.join(arm_dir, name, "sr_manifest.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        out = json.load(f).get("outputs", {}).get(m, {})
    cs = out.get("channel_spread")
    return cs["mean_255"] if cs else None


def score(cols, is_normal):
    entry = {"vs_gt": post.full_reference(cols),
             "grad": {c: post.sharpness(a)[0] for c, a in cols.items()}}
    if is_normal:
        entry["angular_err_deg"] = {c: angular_error_deg(a, cols["gt"])
                                    for c, a in cols.items() if c != "gt"}
    return entry


def process(mat_dir, arms, args):
    name = os.path.basename(mat_dir.rstrip("/"))
    run_dir = newest_run(mat_dir, name, args.run)
    if not run_dir:
        print("  %-28s SKIP (no MatUp output)" % name)
        return None, {}

    lrs, gts, opts = {}, {}, {}
    for m, (src_name, opt_name) in post.MAPS.items():
        lrs[m] = post.read(os.path.join(mat_dir, "SVBRDF", src_name))
        gts[m] = post.read(os.path.join(mat_dir, "GT", src_name))
        p = os.path.join(run_dir, opt_name)
        if os.path.exists(p):
            opts[m] = post.read(p)

    # Same border rule as matup_postprocess.process: measured from MatUp's own
    # zero rim, never from normal (rim is mid-grey) or the pre-cropped maps.
    border, _ = post.detect_border(
        [a for m, a in opts.items() if m != "normal" and m not in post.PRECROPPED])
    scale = gts["baseColor"].shape[0] // lrs["baseColor"].shape[0]
    side = gts["baseColor"].shape[0] - 2 * border

    report = {"material": name, "run": os.path.basename(run_dir), "border": border,
              "size": side, "maps": {}}
    sheets = {}
    for m in post.MAPS:
        if m in opts:
            cols = post.build_columns(lrs[m], gts[m], opts[m], border, scale,
                                      m == "normal", m in post.PRECROPPED)
        else:
            # No MatUp output for this map (height before matup_height.py ran):
            # build the shared columns against GT and drop the MatUp one.
            cols = post.build_columns(lrs[m], gts[m], gts[m], border, scale, False)
            del cols["matup"]
        for label, d in arms:
            p = os.path.join(d, name, m + ".png")
            if os.path.exists(p):
                cols[arm_col(label)] = post.deeptex_column(p, border, side)
            else:
                print("  ! %s: arm %s has no %s (%s)" % (name, label, m, p))
        order = [c for c in BASE_ORDER + [arm_col(l) for l, _ in arms] + ["gt"] if c in cols]
        cols = {c: cols[c] for c in order}
        entry = score(cols, m == "normal")
        entry["channel_spread_mean_255"] = {arm_col(l): channel_spread(d, name, m) for l, d in arms}
        report["maps"][m] = entry
        sheets[m] = post.make_sheet(cols, "%s  -  %s" % (name, m), order)

    rcols = {}
    for col, fname in post.RENDER.items():
        p = os.path.join(run_dir, fname)
        if not os.path.exists(p):
            continue
        a = post.read(p)
        if col not in post.PRECROPPED_RENDER and a.shape[0] > side:
            a = a[border:-border, border:-border] if border else a
        if a.shape[0] != side:
            a = cv2.resize(a, (side, side), interpolation=cv2.INTER_NEAREST)
        rcols[col] = post.to_u16(a)
    for label, d in arms:
        p = os.path.join(d, name, "render.png")
        if os.path.exists(p):
            rcols[arm_col(label)] = post.deeptex_column(p, border, side)
    if "gt" in rcols:
        order = [c for c in ["lr", "bilinear", "matup"] + [arm_col(l) for l, _ in arms] + ["gt"]
                 if c in rcols]
        rcols = {c: rcols[c] for c in order}
        report["maps"]["render"] = score(rcols, False)
        sheets["render"] = post.make_sheet(rcols, "%s  -  render" % name, order)
    return report, sheets


def mean_table(reports, m, cols):
    """Per column: mean PSNR / SSIM over materials, plus exact reproductions."""
    rows = {}
    for c in cols:
        ps, ss, ang, ident, gr, cs = [], [], [], 0, [], []
        for r in reports:
            e = r["maps"].get(m)
            if not e or c not in e["vs_gt"]:
                continue
            v = e["vs_gt"][c]
            if v.get("psnr") is not None:
                ps.append(v["psnr"])
            elif v.get("identical"):
                ident += 1
            if v.get("ssim") is not None:
                ss.append(v["ssim"])
            if "angular_err_deg" in e and c in e["angular_err_deg"]:
                ang.append(e["angular_err_deg"][c])
            gr.append(e["grad"][c])
            s = e.get("channel_spread_mean_255", {}).get(c)
            if s is not None:
                cs.append(s)
        if not (ps or ident):
            continue
        rows[c] = {"n": len(ps) + ident, "psnr": float(np.mean(ps)) if ps else None,
                   "identical": ident, "ssim": float(np.mean(ss)) if ss else None,
                   "angular_err_deg": float(np.mean(ang)) if ang else None,
                   "grad": float(np.mean(gr)) if gr else None,
                   "channel_spread": float(np.mean(cs)) if cs else None}
    gt_grad = [r["maps"][m]["grad"]["gt"] for r in reports if m in r["maps"]]
    return rows, (float(np.mean(gt_grad)) if gt_grad else None)


def write_summary(path, reports, arms):
    cols = BASE_ORDER + [arm_col(l) for l, _ in arms]
    f = lambda v, fmt: "-" if v is None else fmt % v
    lines = ["# DeepTextures arms vs MatUp on the %d MatSynth evaluation materials" % len(reports),
             "",
             "Means over materials. PSNR/SSIM at 8 bits against GT, after the same 8 px crop as",
             "runs/montages. `=GT` counts materials a column reproduces exactly (infinite PSNR,",
             "left out of the PSNR mean). Arms: " +
             ", ".join("`%s` = %s" % (arm_col(l), d) for l, d in arms) + ".", ""]
    tables = {}
    for m in list(post.MAPS) + ["render"]:
        rows, gt_grad = mean_table(reports, m, cols)
        tables[m] = {"columns": rows, "gt_grad": gt_grad}
        normal = m == "normal"
        scalar = m in ("roughness", "metallic", "height")
        names = (["column", "n", "PSNR", "=GT", "SSIM"] + (["ang. err °"] if normal else [])
                 + ["\\|grad\\|"] + (["ch. spread /255"] if scalar else []))
        lines += ["## %s" % m, "", "| " + " | ".join(names) + " |",
                  "|" + "---|" * len(names)]
        for c, r in rows.items():
            line = "| %s | %d | %s | %d | %s |" % (c, r["n"], f(r["psnr"], "%.2f"), r["identical"],
                                                  f(r["ssim"], "%.3f"))
            if normal:
                line += " %s |" % f(r["angular_err_deg"], "%.2f")
            line += " %s |" % f(r["grad"], "%.4f")
            if scalar:
                line += " %s |" % f(r["channel_spread"], "%.2f")
            lines.append(line)
        lines += ["", "GT |grad| %s" % f(gt_grad, "%.4f"), ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return tables


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/runs/matsynth")
    ap.add_argument("--arm", action="append", required=True,
                    help="label=dir of a deeptex_compare.py --out; repeat per arm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--material", default="")
    ap.add_argument("--run", default="", help="MatUp output stamp; default is the newest")
    args = ap.parse_args()

    arms = []
    for a in args.arm:
        label, _, d = a.partition("=")
        if not label or not d or not os.path.isdir(d):
            raise SystemExit("--arm wants label=existing_dir, got %r" % a)
        arms.append((label, d))

    mats = [os.path.join(args.root, args.material)] if args.material else \
        sorted(d for d in glob.glob(os.path.join(args.root, "*"))
               if os.path.isdir(os.path.join(d, "SVBRDF")))
    os.makedirs(os.path.join(args.out, "sheets"), exist_ok=True)

    reports, per_map = [], {}
    for d in mats:
        rep, sheets = process(d, arms, args)
        if not rep:
            continue
        reports.append(rep)
        for m, sheet in sheets.items():
            post.write(os.path.join(args.out, "sheets", "%s_%s.png" % (rep["material"], m)), sheet)
            per_map.setdefault(m, []).append(sheet)
        print("  %-28s %s" % (rep["material"], ", ".join(rep["maps"])))
    if not reports:
        raise SystemExit("nothing scored")

    for m, sheets in per_map.items():
        w = max(s.shape[1] for s in sheets)
        padded = [np.pad(s, ((0, 6), (0, w - s.shape[1]), (0, 0))) for s in sheets]
        big = np.vstack(padded)
        post.write(os.path.join(args.out, "overview_%s.png" % m),
                   cv2.resize(big, (big.shape[1] // 2, big.shape[0] // 2),
                              interpolation=cv2.INTER_AREA))

    tables = write_summary(os.path.join(args.out, "summary.md"), reports, arms)
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump({"arms": dict(arms), "means": tables, "materials": reports}, fh, indent=1)
    print("%d material(s) -> %s" % (len(reports), os.path.join(args.out, "summary.md")))


if __name__ == "__main__":
    main()
