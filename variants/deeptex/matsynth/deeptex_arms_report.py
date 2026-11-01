#!/usr/bin/env python3
"""Score several DeepTextures arms against MatUp and GT on every map and the render.

Run it in the dataprep container, after deeptex_compare.py --maps all has written
each arm's outputs:

    docker compose run --rm dataprep python variants/deeptex/matsynth/deeptex_arms_report.py \\
        --arm textures_db=/workspace/runs/deeptex/eval/paper_allmaps \\
        --arm matsynth_all=/workspace/runs/deeptex/eval/matsynth_all \\
        --out /workspace/runs/deeptex/report_matsynth_all

Written under --out (and nowhere else):

    montages/<material>.png        rows = method, columns = five maps + render
    overview_<map>.png             rows = the 12 materials, columns = method
    sheets/<material>_<map>.png    one map at full 496^2 resolution, for zooming in
    summary.md                     means over the materials, one table per map, and
                                   head-to-head counts between the arms
    per_material.md                PSNR / SSIM (and angle for normal) per material
    metrics_long.csv               one row per material x map x column - for pivots
    report.json                    every number, per material

The montage and overview follow matup_montage.py and matup_overview.py - same
cell rendering (their to_display, imported), same layout - so the new sheets sit
next to the committed ones without looking like a different study. Nothing is
normalised for display: an all-zero metallic is black, as there.

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
* |grad| (matup_postprocess.sharpness), reference-free, as in the committed JSON,
  also as a percentage of the GT's: PSNR rewards blur, and this says whether a
  column has roughly the detail the GT has (commit 11fecc7 for why it matters).
* normal: mean angular error in degrees, also on 8-bit-rounded inputs. PSNR on a
  normal map scores pixels, not directions.
* channel_spread from each arm's sr_manifest.json: how far apart the network's
  three output channels were on a 1-channel map, measured on the RAW output
  before clipping. Trained on replicated grey maps, the MatSynth arm should push
  this toward 0; the textures_db arm never saw them.

A metallic GT is a constant 0 on 9 of the 12 materials; a column that reproduces
it exactly has infinite PSNR. Means are taken over finite values and the number
of exact reproductions is reported beside them. GPU inference is not
bit-reproducible here (DEEPTEX-DEVIATIONS 0.9): a rerun moves outputs by up to
1 LSB, which can move a PSNR in the second decimal.
"""
import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.util import paths  # noqa: E402
from variants.compare import matup_postprocess as post  # noqa: E402
from variants.compare import matup_montage as mont  # noqa: E402
from variants.compare import matup_overview as ov  # noqa: E402
from variants.compare.matup_runs import newest_run  # noqa: E402  (was a local copy, :104-109)

BASE_ORDER = ["lr", "bilinear", "matup"]
MAP_ORDER = list(post.MAPS) + ["render"]
CELL = 320            # montage cell, as matup_montage's default --size
OV_CELL = 190         # overview cell, as matup_overview's default --size
FONT = cv2.FONT_HERSHEY_SIMPLEX
NAMES = {"lr": "LR", "bilinear": "bilinear", "matup": "MatUp", "gt": "GT"}


def arm_col(label):
    return "dt_" + label


def label_lines(col):
    """Display label, one or two lines. Arms read as the paper's method name with
    the training data in brackets - "dt_matsynth_all" means nothing to a reader
    (same reasoning as commit 9f6b8be)."""
    if col.startswith("dt_"):
        return ["DeepTextures", "(%s)" % col[3:]]
    return [NAMES.get(col, col)]


def angular_error_deg(pred, gt):
    """Mean angle between decoded normals, both rounded to 8 bits first."""
    def vec(a):
        v = np.rint(post.to_float(a) * 255.0) / 255.0 * 2.0 - 1.0
        n = np.linalg.norm(v, axis=2, keepdims=True)
        return v / np.maximum(n, 1e-8)
    d = np.clip((vec(pred) * vec(gt)).sum(axis=2), -1.0, 1.0)
    return float(np.degrees(np.arccos(d)).mean())


def channel_spread(arm_dir, name, m):
    p = os.path.join(arm_dir, name, "sr_manifest.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        out = json.load(f).get("outputs", {}).get(m, {})
    cs = out.get("channel_spread")
    return cs["mean_255"] if cs else None


def offset_diagnostics(pred, gt):
    """(per-channel mean offset vs GT in 8-bit levels, PSNR once it is removed).

    Diagnostic, not a headline metric. The paper's recipe (no pre-training, pixel
    weight 0.01, a saturated discriminator, a VGG loss that is nearly blind to a
    constant shift) lets the generator drift in colour: measured on its own
    training pairs the MatSynth checkpoint shifts normals by about (-20, -7, -6)
    levels. That offset alone costs a near-flat normal map tens of dB, so this
    separates "wrong everywhere by a constant" from "wrong structure".
    """
    p = np.rint(post.to_float(pred) * 255.0)
    g = np.rint(post.to_float(gt) * 255.0)
    if p.ndim == 2:
        p, g = p[..., None], g[..., None]
    off = (p - g).reshape(-1, p.shape[-1]).mean(axis=0)
    mse = ((np.clip(p - off, 0, 255) - g) ** 2).mean()
    psnr = None if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))
    # The arrays are cv2 order; report colour offsets as RGB like everything a
    # reader compares them with.
    return [float(o) for o in (off[::-1] if off.size == 3 else off)], psnr


def score(cols, is_normal):
    grad = {c: post.sharpness(a)[0] for c, a in cols.items()}
    diag = {c: offset_diagnostics(a, cols["gt"]) for c, a in cols.items() if c != "gt"}
    entry = {"vs_gt": post.full_reference(cols), "grad": grad,
             "grad_vs_gt": {c: (g / grad["gt"] if grad["gt"] else None)
                            for c, g in grad.items() if c != "gt"},
             "mean_offset_255": {c: d[0] for c, d in diag.items()},
             "psnr_offset_removed": {c: d[1] for c, d in diag.items()}}
    if is_normal:
        entry["angular_err_deg"] = {c: angular_error_deg(a, cols["gt"])
                                    for c, a in cols.items() if c != "gt"}
    return entry


def process(mat_dir, arms, args):
    """One material -> (report, {map: full-res sheet}, {map: {column: display cell}})."""
    name = os.path.basename(mat_dir.rstrip("/"))
    run_dir = newest_run(mat_dir, name, args.run)
    if not run_dir:
        print("  %-28s SKIP (no MatUp output)" % name)
        return None, {}, {}

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
    sheets, cells = {}, {}
    arm_cols = [arm_col(l) for l, _ in arms]
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
        order = [c for c in BASE_ORDER + arm_cols + ["gt"] if c in cols]
        cols = {c: cols[c] for c in order}
        entry = score(cols, m == "normal")
        entry["channel_spread_mean_255"] = {arm_col(l): channel_spread(d, name, m) for l, d in arms}
        report["maps"][m] = entry
        sheets[m] = post.make_sheet(cols, "%s  -  %s" % (name, m), order)
        # Downscaled now, so twelve materials x six maps x six columns of 496^2
        # 16-bit arrays never have to be held at once.
        cells[m] = {c: mont.to_display(a, CELL) for c, a in cols.items()}

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
        order = [c for c in BASE_ORDER + arm_cols + ["gt"] if c in rcols]
        rcols = {c: rcols[c] for c in order}
        report["maps"]["render"] = score(rcols, False)
        sheets["render"] = post.make_sheet(rcols, "%s  -  render" % name, order)
        cells["render"] = {c: mont.to_display(a, CELL) for c, a in rcols.items()}
    return report, sheets, cells


# ------------------------------------------------------------------ pictures

def draw_lines(img, lines, x, y, width, start, colour):
    """Draw 1-2 label lines with their top-left near (x, y), shrunk to `width`."""
    for i, t in enumerate(lines):
        s = ov.fit(t, width, start if i == 0 else start * 0.85)
        cv2.putText(img, t, (x, y + i * int(26 * start + 6)), FONT, s, colour, 1, cv2.LINE_AA)


def write_montage(path, title, cells, row_keys):
    """matup_montage.build's layout: methods down, maps across."""
    pad, head, top = mont.PAD, mont.HEAD, mont.TITLE
    left = 170
    w = left + len(MAP_ORDER) * (CELL + pad) + pad
    h = top + head + len(row_keys) * (CELL + pad) + pad
    sheet = np.full((h, w, 3), 18, np.uint8)
    cv2.putText(sheet, title, (10, 22), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for c, m in enumerate(MAP_ORDER):
        x = left + c * (CELL + pad)
        cv2.putText(sheet, m, (x + 4, top + 22), FONT, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    for r, key in enumerate(row_keys):
        y = top + head + r * (CELL + pad)
        draw_lines(sheet, label_lines(key), 10, y + CELL // 2 - 8, left - 18, 0.62,
                   (210, 210, 210))
        for c, m in enumerate(MAP_ORDER):
            x = left + c * (CELL + pad)
            cell = cells.get(m, {}).get(key)
            if cell is None:
                note = ["run matup_height.py"] if (key, m) == ("matup", "height") else []
                cell = mont.placeholder(CELL, note)
            sheet[y:y + CELL, x:x + CELL] = cell
    post.write(path, sheet)


def write_overview(path, m, rows, col_keys):
    """matup_overview.main's layout: materials down, methods across."""
    pad, left, top, head = ov.PAD, ov.LEFT, ov.TITLE, 46
    w = left + len(col_keys) * (OV_CELL + pad) + pad
    h = top + head + len(rows) * (OV_CELL + pad) + pad
    sheet = np.full((h, w, 3), 18, np.uint8)
    cv2.putText(sheet, "MatUp vs DeepTextures arms  -  %s  -  %d materials" % (m, len(rows)),
                (10, 24), FONT, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    for c, key in enumerate(col_keys):
        x = left + c * (OV_CELL + pad)
        draw_lines(sheet, label_lines(key), x + 3, top + 18, OV_CELL - 6, 0.5, (215, 215, 215))
    for r, (name, cells) in enumerate(rows):
        y = top + head + r * (OV_CELL + pad)
        cv2.putText(sheet, name, (8, y + OV_CELL // 2), FONT,
                    ov.fit(name, left - 16, 0.46), (215, 215, 215), 1, cv2.LINE_AA)
        for c, key in enumerate(col_keys):
            x = left + c * (OV_CELL + pad)
            cell = cells.get(key)
            if cell is None:
                cell = np.full((OV_CELL, OV_CELL, 3), 30, np.uint8)
                cv2.putText(cell, "n/a", (OV_CELL // 2 - 18, OV_CELL // 2), FONT, 0.5,
                            (110, 110, 110), 1, cv2.LINE_AA)
            else:
                cell = cv2.resize(cell, (OV_CELL, OV_CELL), interpolation=cv2.INTER_AREA)
            sheet[y:y + OV_CELL, x:x + OV_CELL] = cell
    post.write(path, sheet)


# ------------------------------------------------------------------ numbers

def common_materials(reports, m, cols):
    """Materials on which EVERY column has a finite PSNR.

    Means are taken over this set only. Otherwise a column that reproduces GT
    exactly on nine materials (metallic: an all-zero map in, an all-zero map out)
    would be averaged over three materials while its neighbours average over
    twelve, and the two means would not describe the same thing.
    """
    keep = []
    for r in reports:
        e = r["maps"].get(m)
        if e and all(c in e["vs_gt"] and e["vs_gt"][c].get("psnr") is not None for c in cols):
            keep.append(r["material"])
    return set(keep)


def mean_table(reports, m, cols):
    """Per column: means over the common materials, plus exact reproductions over all."""
    rows = {}
    present = [c for c in cols if any(m in r["maps"] and c in r["maps"][m]["vs_gt"]
                                      for r in reports)]
    common = common_materials(reports, m, present)
    for c in present:
        ps, ss, ang, ident, gr, gp, cs, po, off = [], [], [], 0, [], [], [], [], []
        for r in reports:
            e = r["maps"].get(m)
            if not e or c not in e["vs_gt"]:
                continue
            v = e["vs_gt"][c]
            if v.get("identical"):
                ident += 1
            if r["material"] not in common:
                continue
            ps.append(v["psnr"])
            if v.get("ssim") is not None:
                ss.append(v["ssim"])
            if e["psnr_offset_removed"].get(c) is not None:
                po.append(e["psnr_offset_removed"][c])
            off.append(max(abs(o) for o in e["mean_offset_255"][c]))
            if "angular_err_deg" in e and c in e["angular_err_deg"]:
                ang.append(e["angular_err_deg"][c])
            gr.append(e["grad"][c])
            if e["grad_vs_gt"].get(c) is not None:
                gp.append(100.0 * e["grad_vs_gt"][c])
            s = e.get("channel_spread_mean_255", {}).get(c)
            if s is not None:
                cs.append(s)
        mean = lambda xs: float(np.mean(xs)) if xs else None
        rows[c] = {"n": len(ps), "psnr": mean(ps), "identical": ident, "ssim": mean(ss),
                   "psnr_offset_removed": mean(po), "max_abs_offset_255": mean(off),
                   "angular_err_deg": mean(ang), "grad": mean(gr),
                   "grad_pct_gt": mean(gp), "channel_spread": mean(cs)}
    gt_grad = [r["maps"][m]["grad"]["gt"] for r in reports
               if m in r["maps"] and r["material"] in common]
    return rows, (float(np.mean(gt_grad)) if gt_grad else None)


def head_to_head(reports, m, a, b):
    """On how many materials column a beats column b. An exact reproduction of GT
    (infinite PSNR) counts as the best possible PSNR, not as missing."""
    def psnr(v):
        return float("inf") if v.get("identical") else v.get("psnr")
    n = psnr_w = ssim_w = ang_w = ang_n = 0
    for r in reports:
        e = r["maps"].get(m)
        if not e or a not in e["vs_gt"] or b not in e["vs_gt"]:
            continue
        pa, pb = psnr(e["vs_gt"][a]), psnr(e["vs_gt"][b])
        if pa is None or pb is None:
            continue
        n += 1
        psnr_w += pa > pb
        ssim_w += (e["vs_gt"][a].get("ssim") or 0) > (e["vs_gt"][b].get("ssim") or 0)
        if "angular_err_deg" in e:
            ang_n += 1
            ang_w += e["angular_err_deg"][a] < e["angular_err_deg"][b]
    return {"n": n, "psnr": psnr_w, "ssim": ssim_w, "angle": ang_w if ang_n else None}


def write_summary(path, reports, arms):
    cols = BASE_ORDER + [arm_col(l) for l, _ in arms]
    f = lambda v, fmt: "-" if v is None else fmt % v
    lines = ["# DeepTextures arms vs MatUp on the %d MatSynth evaluation materials" % len(reports),
             "",
             "PSNR/SSIM at 8 bits against GT, after the same 8 px crop as runs/montages.",
             "",
             "- `n`: the materials every column has a finite PSNR on. All means in a table are",
             "  over this same set, so the rows are comparable.",
             "- `=GT`: materials a column reproduces exactly (infinite PSNR), counted over all",
             "  materials. Metallic is the case that needs it: 9 of the 12 GTs are all zero.",
             "- `offset`: mean over materials of the largest per-channel mean offset vs GT, in",
             "  8-bit levels. `PSNR -off` is PSNR after removing that offset. Both are",
             "  DIAGNOSTICS: they show how much of a column's error is a constant shift (the",
             "  DeepTextures recipe drifts in colour) rather than wrong structure.",
             "- `|grad| %GT`: each material's gradient magnitude as a percentage of its GT's,",
             "  averaged. 100 means as much detail as the GT; PSNR rewards blur, this does not.",
             "- Head-to-head lines count materials where the first column wins; an exact",
             "  reproduction of GT counts as the best PSNR.",
             "",
             "Arms: " + ", ".join("`%s` = %s" % (arm_col(l), d) for l, d in arms) + ".", ""]
    tables, h2h = {}, {}
    pairs = []
    if len(arms) >= 2:
        pairs.append((arm_col(arms[-1][0]), arm_col(arms[-2][0])))
    for l, _ in arms:
        pairs.append((arm_col(l), "matup"))
    for m in MAP_ORDER:
        rows, gt_grad = mean_table(reports, m, cols)
        tables[m] = {"columns": rows, "gt_grad": gt_grad}
        normal = m == "normal"
        scalar = m in ("roughness", "metallic", "height")
        names = (["column", "n", "PSNR", "=GT", "SSIM"] + (["ang. err °"] if normal else [])
                 + ["offset", "PSNR -off", "\\|grad\\| %GT"]
                 + (["ch. spread /255"] if scalar else []))
        lines += ["## %s" % m, "", "| " + " | ".join(names) + " |",
                  "|" + "---|" * len(names)]
        for c, r in rows.items():
            line = "| %s | %d | %s | %d | %s |" % (c, r["n"], f(r["psnr"], "%.2f"), r["identical"],
                                                  f(r["ssim"], "%.3f"))
            if normal:
                line += " %s |" % f(r["angular_err_deg"], "%.2f")
            line += " %s | %s |" % (f(r["max_abs_offset_255"], "%.1f"),
                                    f(r["psnr_offset_removed"], "%.2f"))
            line += " %s |" % f(r["grad_pct_gt"], "%.0f%%")
            if scalar:
                line += " %s |" % f(r["channel_spread"], "%.2f")
            lines.append(line)
        lines += [""]
        h2h[m] = {}
        for a, b in pairs:
            hh = head_to_head(reports, m, a, b)
            h2h[m]["%s_vs_%s" % (a, b)] = hh
            if hh["n"]:
                txt = "- `%s` beats `%s` on PSNR %d/%d, SSIM %d/%d" % (
                    a, b, hh["psnr"], hh["n"], hh["ssim"], hh["n"])
                if hh["angle"] is not None:
                    txt += ", angle %d/%d" % (hh["angle"], hh["n"])
                lines.append(txt)
        lines += [""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return tables, h2h


def write_per_material(path, reports, arms):
    cols = BASE_ORDER + [arm_col(l) for l, _ in arms]
    lines = ["# Per-material numbers", "",
             "PSNR / SSIM against GT at 8 bits; `=GT` is an exact reproduction. For normal,",
             "the mean angular error in degrees follows after a slash.", ""]
    for m in MAP_ORDER:
        present = [c for c in cols if any(m in r["maps"] and c in r["maps"][m]["vs_gt"]
                                         for r in reports)]
        lines += ["## %s" % m, "", "| material | " + " | ".join(present) + " |",
                  "|---|" + "---|" * len(present)]
        for r in reports:
            e = r["maps"].get(m)
            if not e:
                continue
            cells = []
            for c in present:
                v = e["vs_gt"].get(c)
                if v is None:
                    cells.append("-")
                    continue
                p = "=GT" if v.get("identical") else ("-" if v.get("psnr") is None
                                                     else "%.2f" % v["psnr"])
                cell = "%s / %.3f" % (p, v["ssim"]) if v.get("ssim") is not None else p
                if "angular_err_deg" in e and c in e["angular_err_deg"]:
                    cell += " / %.1f°" % e["angular_err_deg"][c]
                cells.append(cell)
            lines.append("| %s | %s |" % (r["material"], " | ".join(cells)))
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_csv(path, reports):
    fields = ["material", "map", "column", "psnr", "identical", "ssim", "grad",
              "grad_pct_gt", "angular_err_deg", "channel_spread_mean_255",
              "mean_offset_255", "psnr_offset_removed"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in reports:
            for m, e in r["maps"].items():
                for c in list(e["vs_gt"]) + ["gt"]:
                    v = e["vs_gt"].get(c, {})
                    gp = e["grad_vs_gt"].get(c)
                    w.writerow({
                        "material": r["material"], "map": m, "column": c,
                        "psnr": v.get("psnr"), "identical": v.get("identical", False),
                        "ssim": v.get("ssim"), "grad": e["grad"][c],
                        "grad_pct_gt": None if gp is None else 100.0 * gp,
                        "angular_err_deg": e.get("angular_err_deg", {}).get(c),
                        "channel_spread_mean_255": e.get("channel_spread_mean_255", {}).get(c),
                        # per-channel (RGB) offset as "r;g;b", so the cell stays one field
                        "mean_offset_255": None if c == "gt" else
                        ";".join("%.2f" % o for o in e["mean_offset_255"][c]),
                        "psnr_offset_removed": e["psnr_offset_removed"].get(c),
                    })


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
    row_keys = BASE_ORDER + [arm_col(l) for l, _ in arms] + ["gt"]

    mats = [os.path.join(args.root, args.material)] if args.material else \
        sorted(d for d in glob.glob(os.path.join(args.root, "*"))
               if os.path.isdir(os.path.join(d, "SVBRDF")))
    for sub in ("sheets", "montages"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    reports, per_map = [], {}
    for d in mats:
        rep, sheets, cells = process(d, arms, args)
        if not rep:
            continue
        reports.append(rep)
        name = rep["material"]
        for m, sheet in sheets.items():
            post.write(os.path.join(args.out, "sheets", "%s_%s.png" % (name, m)), sheet)
        write_montage(os.path.join(args.out, "montages", name + ".png"),
                      "%s    MatUp %s" % (name, rep["run"]), cells, row_keys)
        for m, c in cells.items():
            per_map.setdefault(m, []).append((name, c))
        print("  %-28s %s" % (name, ", ".join(rep["maps"])))
    if not reports:
        raise SystemExit("nothing scored")

    for m, rows in per_map.items():
        write_overview(os.path.join(args.out, "overview_%s.png" % m), m, rows, row_keys)

    tables, h2h = write_summary(os.path.join(args.out, "summary.md"), reports, arms)
    write_per_material(os.path.join(args.out, "per_material.md"), reports, arms)
    write_csv(os.path.join(args.out, "metrics_long.csv"), reports)
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump({"arms": dict(arms), "means": tables, "head_to_head": h2h,
                   "materials": reports}, fh, indent=1)
    print("%d material(s) -> %s" % (len(reports), os.path.join(args.out, "summary.md")))


if __name__ == "__main__":
    main()
