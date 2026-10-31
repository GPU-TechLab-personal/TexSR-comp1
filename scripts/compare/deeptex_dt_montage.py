#!/usr/bin/env python3
"""DeepTextures-only contact sheets: the arms against each other and GT, no baselines.

Runs in the matup container (torch + MatUp's renderer + cv2), from the repo root:

    docker compose -f docker/compose.yaml run --rm matup python \\
        /workspace/scripts/compare/deeptex_dt_montage.py \\
        --arm matsynth_all=/workspace/runs/deeptex/eval/matsynth_all \\
        --arm render_r1_40k=/workspace/runs/deeptex/eval/render_r1_40k \\
        --out /workspace/runs/montages/deeptex_render_loss/dt_only

The container starts in /workspace/external/matup_egsr, so the script path and
--out must be absolute; an --out under external/ is refused (exit 2).

WHY THIS EXISTS
---------------
deeptex_arms_report.py's <material>.png and svbrdf_render_eval.py's
render_<material>.png both carry LR / bilinear / MatUp rows, and with ten
DeepTextures checkpoints beside them the differences BETWEEN the checkpoints -
the question the render-loss arms (R0a / R0b / R1) ask - are what is hardest to
see. These sheets keep only the --arm rows (in the order given) and GT.

Nothing is recomputed differently from those two scripts; their code is imported:

  maps     deeptex_arms_report.process: the border post.detect_border measures on
           MatUp's _opt maps (normal and the pre-cropped height excluded), the DT
           PNG through post.deeptex_column, the GT through the same crop and
           post.to_u16, displayed with matup_montage.to_display (>> 8, no
           per-cell normalisation: an all-zero metallic is black; the normal is
           its stored OpenGL RGB encoding; grey maps stay grey).
  renders  svbrdf_render_eval.process: load_dt / load_gt (load_set + position3d of
           the set's own height), prepare (crop, get_renorm_clamped_maps),
           render_display / eval_image (MatUp's torch renderer, UNGUARDED),
           build_lights (EVAL_LIGHT and the 'grazing' sheet light), to_cell and
           heat_cell at svbrdf_render_eval.HEAT_GAIN. The grazing light is rendered
           in the same --chunk of held-out lights svbrdf_render_eval renders it
           in, so every render cell is what render_<material>.png shows (at a
           different cell size). The GT row's EVAL_LIGHT render is self-checked
           against MatUp's committed render_gt.png (svbrdf_render_eval.self_check,
           exit 3 on failure).

OUTPUTS (all in --out; earlier outputs of these names are removed first)
  dt_<material>.png        rows = the arms + GT; columns = baseColor | normal |
                           roughness | metallic | height | render EVAL_LIGHT |
                           render held-out grazing | |pred - GT| x HEAT_GAIN (EVAL)
  dt_overview_<key>.png    rows = materials, columns = the arms + GT, for key in
                           baseColor, normal, roughness, metallic, height, render
                           (render = EVAL_LIGHT render of the predicted SVBRDF)
  README.md                rows -> tags -> checkpoints, columns, crop, gain, sizes,
                           and the exact command that regenerates the folder
"""
import argparse
import datetime
import glob
import math
import os
import sys

# Before anything from external/ is imported: Python 3.10 must not leave
# cpython-310.pyc files inside the submodule (svbrdf_render_eval does the same).
sys.dont_write_bytecode = True

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MATUP_ROOT = os.environ.get("MATUP_ROOT", "/workspace/external/matup_egsr")
for _p in (MATUP_ROOT, os.path.join(HERE, os.pardir, "matup"),
           os.path.join(HERE, os.pardir, "common"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from matup.renderer import Renderer  # noqa: E402

import matup_montage as mont  # noqa: E402
import matup_overview as ov  # noqa: E402
import matup_postprocess as post  # noqa: E402
import matup_render as mr  # noqa: E402
import svbrdf_render_eval as ev  # noqa: E402

SCRIPT = "/workspace/scripts/compare/deeptex_dt_montage.py"
DEFAULT_ROOT = "/workspace/runs/matsynth"
DEFAULT_CELL = 224
DEFAULT_OV_CELL = 150
DEFAULT_CHUNK = 10               # svbrdf_render_eval --chunk default

MAP_ORDER = ev.MAP_ORDER         # baseColor, normal, roughness, metallic, height
OV_KEYS = list(MAP_ORDER) + ["render"]
FONT = ev.FONT
LEFT = ev.SHEET_LEFT             # row-label gutter of the per-material sheet
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 9]   # lossless; only smaller files
SEP = (95, 95, 95)               # separator between arm groups and before GT


def die(code, msg):
    print("[fatal] %s" % msg)
    sys.exit(code)


def write_png(path, img):
    """post.write with maximum (lossless) PNG compression."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, img, PNG_PARAMS):
        die(2, "cv2.imwrite failed for " + path)


def mb(path):
    return os.path.getsize(path) / 1e6


# -------------------------------------------------------------------- labels

def text_w(t, scale):
    return cv2.getTextSize(t, FONT, scale, 1)[0][0]


def wrap_part(part, width, scale):
    """One label part as few lines as fit `width`: whole, else the most balanced
    two-line split that fits ("DeepTextures +" / "render loss", not "...render" /
    "loss"), else greedy."""
    words = part.split(" ")
    if text_w(part, scale) <= width or len(words) == 1:
        return [part]
    splits = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
    fits = [s for s in splits if max(text_w(s[0], scale), text_w(s[1], scale)) <= width]
    if fits:
        return list(min(fits, key=lambda s: max(text_w(s[0], scale), text_w(s[1], scale))))
    lines, cur = [], ""
    for word in words:
        t = (cur + " " + word).strip()
        if cur and text_w(t, scale) > width:
            lines.append(cur)
            cur = word
        else:
            cur = t
    return lines + ([cur] if cur else [])


def wrap_label(text, width, scale):
    """ev.label_lines (break before the bracket), then each part wrapped to
    `width` at `scale` (wrap_part); a single word that still does not fit is
    shrunk when drawn (ov.fit)."""
    lines = []
    for part in ev.label_lines(text):
        lines += wrap_part(part, width, scale)
    return lines


def put_lines(img, lines, x, y, width, scale, colour, step):
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x, y + i * step), FONT, ov.fit(t, width, scale),
                    colour, 1, cv2.LINE_AA)


def gt_heat_cell(size):
    """The GT row has no error to show: a black cell that says so."""
    cell = np.zeros((size, size, 3), np.uint8)
    for i, t in enumerate(("GT = reference", "no |pred - GT|")):
        s = ov.fit(t, size - 16, 0.5)
        w = cv2.getTextSize(t, FONT, s, 1)[0][0]
        cv2.putText(cell, t, (max(4, size // 2 - w // 2), size // 2 - 4 + 22 * i), FONT, s,
                    (150, 150, 150), 1, cv2.LINE_AA)
    return cell


# ------------------------------------------------------------------ rendering

def held_light_image(rmaps, held, j, chunk, where):
    """Held-out light j rendered inside the same chunk of held-out lights that
    svbrdf_render_eval.render_chunks / score render it in, then taken out exactly
    as their `kept[j]` is."""
    lo = (j // chunk) * chunk
    hi = min(lo + chunk, held.shape[0])
    d = ev.render_display(rmaps, held[lo:hi], where)
    h, w = rmaps[0].shape[:2]
    return d[:, :, j - lo].cpu().numpy().reshape(h, w, 3)


def render_pair(maps, border, lights, chunk, device, where):
    """(EVAL_LIGHT render, grazing held-out render, n_z == 0 texels fixed)."""
    rmaps, fix = ev.prepare(maps, border, device)
    img = ev.eval_image(rmaps, lights["eval"], where + " eval")
    graz = held_light_image(rmaps, lights["held"], lights["sheet"]["grazing"], chunk,
                            where + " grazing")
    del rmaps
    return img, graz, fix


def map_display(a, size):
    """The montage cell of a stored map array (mont.to_display, as
    deeptex_arms_report does); a 4th channel, if any, is not drawn."""
    return mont.to_display(a[..., :3] if a.ndim == 3 else a, size)


# --------------------------------------------------------------- per material

def process(mat_dir, cols, args, device, lights):
    """One material -> (info, [(label, [cells])] sheet rows, {key: {col: cell}})."""
    name = os.path.basename(mat_dir.rstrip("/"))
    run_dir = ev.newest_run(mat_dir, name, args.run)
    if not run_dir:
        print("  %-28s SKIP (no MatUp output)" % name)
        return None, None, None
    ev.require_readable([os.path.join(mat_dir, "GT", m + ".png") for m in MAP_ORDER]
                        + [os.path.join(run_dir, m + "_opt.png")
                           for m in MAP_ORDER if m != "height"], name)

    # deeptex_arms_report.py:166-177 / svbrdf_render_eval.process: MatUp's zero
    # rim, never measured on normal (mid-grey rim) or the pre-cropped height.
    opts = {}
    for m, (_, opt_name) in post.MAPS.items():
        p = os.path.join(run_dir, opt_name)
        if os.path.exists(p):
            opts[m] = post.read(p)
    border, sides = post.detect_border(
        [a for m, a in opts.items() if m != "normal" and m not in post.PRECROPPED])
    del opts

    gt_maps, _, _ = ev.load_gt(mat_dir)
    full = gt_maps[0].shape[0]
    side = full - 2 * border
    mask_hw = ev.gt_valid_mask(os.path.join(mat_dir, "GT"), border)
    gt_eval, gt_graz, gt_fix = render_pair(gt_maps, border, lights, args.chunk, device,
                                           "%s gt" % name)
    sc = ev.self_check("gt", gt_eval, run_dir, border, full)

    info = {"material": name, "run": os.path.basename(run_dir), "border": border,
            "border_sides": list(sides), "side": side, "self_check_lsb": sc["max_abs"] * ev.U16,
            "valid_frac": float(mask_hw.mean()), "missing": [], "nz_zero_fixed": {}}
    if gt_fix:
        info["nz_zero_fixed"]["gt"] = gt_fix

    rows, ov_cells = [], {k: {} for k in OV_KEYS}
    for col in cols:
        key = col["key"]
        d = os.path.join(col["dir"], name)
        need = [os.path.join(d, m + ".png") for m in MAP_ORDER]
        lost = [os.path.basename(p) for p in need if not os.path.exists(p)]
        if lost:
            print("  ! %s: arm %s has no %s" % (name, col["tag"], ", ".join(lost)))
            info["missing"].append(col["tag"])
            rows.append((col, [mont.placeholder(args.cell, ["missing"])] * 8))
            continue
        ev.require_readable(need, "%s / %s" % (name, key))

        cells = []
        for m, p in zip(MAP_ORDER, need):
            a = post.deeptex_column(p, border, side)              # deeptex_arms_report.py:197
            cells.append(map_display(a, args.cell))
            ov_cells[m][key] = map_display(a, args.ov_cell)

        maps, _, _ = ev.load_dt(col["dir"], name)
        if maps[0].shape[0] != full or any(t.shape[:2] != maps[0].shape[:2] for t in maps):
            die(2, "%s / %s: map sizes %s, GT is %d" % (
                name, key, [tuple(t.shape) for t in maps], full))
        img, graz, fix = render_pair(maps, border, lights, args.chunk, device,
                                     "%s %s" % (name, key))
        if fix:
            info["nz_zero_fixed"][key] = fix
        cells += [ev.to_cell(img, args.cell), ev.to_cell(graz, args.cell),
                  ev.heat_cell(img, gt_eval, mask_hw, args.cell)]
        ov_cells["render"][key] = ev.to_cell(img, args.ov_cell)
        rows.append((col, cells))

    cells = []
    for m in MAP_ORDER:
        # deeptex_arms_report: post.build_columns' crop_hr(gt) -> to_u16.
        a = post.to_u16(mr.crop(post.read(os.path.join(mat_dir, "GT", m + ".png")), border))
        cells.append(map_display(a, args.cell))
        ov_cells[m]["gt"] = map_display(a, args.ov_cell)
    cells += [ev.to_cell(gt_eval, args.cell), ev.to_cell(gt_graz, args.cell),
              gt_heat_cell(args.cell)]
    ov_cells["render"]["gt"] = ev.to_cell(gt_eval, args.ov_cell)
    rows.append(({"key": "gt", "base": "gt"}, cells))
    return info, rows, ov_cells


# ------------------------------------------------------------------- pictures

def separators(keys_bases):
    """Indices i where a separator goes BEFORE entry i: the arm base changes
    (matsynth_all | R0a | R0b | R1) or GT starts."""
    return [i for i in range(1, len(keys_bases)) if keys_bases[i] != keys_bases[i - 1]]


def write_sheet(path, title, notes, heads, rows, names, cell):
    """svbrdf_render_eval.write_render_sheet's layout (methods down, maps and
    renders across), with two-line column heads and wrapped row labels."""
    pad = mont.PAD
    top = mont.TITLE + 18 * len(notes)
    head_lines = [wrap_label(h, cell - 8, 0.5) if isinstance(h, str) else h for h in heads]
    head = 14 + 19 * max(len(h) for h in head_lines) + 8
    w = LEFT + len(heads) * (cell + pad) + pad
    h = top + head + len(rows) * (cell + pad) + pad
    sheet = np.full((h, w, 3), 18, np.uint8)
    cv2.putText(sheet, title, (10, 22), FONT, ov.fit(title, w - 20, 0.6), (255, 255, 255),
                1, cv2.LINE_AA)
    for i, t in enumerate(notes):
        cv2.putText(sheet, t, (10, 22 + 18 * (i + 1)), FONT, ov.fit(t, w - 20, 0.45),
                    (200, 200, 200), 1, cv2.LINE_AA)
    for c, lines in enumerate(head_lines):
        x = LEFT + c * (cell + pad)
        put_lines(sheet, lines, x + 4, top + 20, cell - 8, 0.5, (210, 210, 210), 19)
    seps = separators([col["base"] for col, _ in rows])
    for r, (col, cells) in enumerate(rows):
        y = top + head + r * (cell + pad)
        if r in seps:
            cv2.line(sheet, (6, y - pad // 2 - 1), (w - 6, y - pad // 2 - 1), SEP, 2)
        lines = wrap_label(names[col["key"]], LEFT - 18, 0.58)
        step = 23
        put_lines(sheet, lines, 10, y + cell // 2 - (len(lines) - 1) * step // 2 + 6,
                  LEFT - 18, 0.58, (210, 210, 210), step)
        for c, cl in enumerate(cells):
            x = LEFT + c * (cell + pad)
            sheet[y:y + cell, x:x + cell] = cl
    write_png(path, sheet)


def write_overview(path, title, rows, col_keys, bases, names, cell):
    """svbrdf_render_eval.write_overview / deeptex_arms_report.write_overview's
    layout (materials down, methods across), with wrapped column heads."""
    pad, left, top = ov.PAD, ov.LEFT, ov.TITLE
    head_lines = [wrap_label(names[k], cell - 6, 0.42) for k in col_keys]
    head = 12 + 16 * max(len(h) for h in head_lines) + 6
    w = left + len(col_keys) * (cell + pad) + pad
    h = top + head + len(rows) * (cell + pad) + pad
    sheet = np.full((h, w, 3), 18, np.uint8)
    cv2.putText(sheet, title, (10, 24), FONT, ov.fit(title, w - 20, 0.62), (255, 255, 255),
                1, cv2.LINE_AA)
    seps = separators(bases)
    for c, lines in enumerate(head_lines):
        x = left + c * (cell + pad)
        if c in seps:
            cv2.line(sheet, (x - pad // 2 - 1, top + 4), (x - pad // 2 - 1, h - 4), SEP, 2)
        put_lines(sheet, lines, x + 3, top + 16, cell - 6, 0.42, (215, 215, 215), 16)
    for r, (name, cells) in enumerate(rows):
        y = top + head + r * (cell + pad)
        cv2.putText(sheet, name, (8, y + cell // 2), FONT,
                    ov.fit(name, left - 16, 0.46), (215, 215, 215), 1, cv2.LINE_AA)
        for c, key in enumerate(col_keys):
            x = left + c * (cell + pad)
            cl = cells.get(key)
            if cl is None:
                cl = np.full((cell, cell, 3), 30, np.uint8)
                cv2.putText(cl, "n/a", (cell // 2 - 18, cell // 2), FONT, 0.5,
                            (110, 110, 110), 1, cv2.LINE_AA)
            sheet[y:y + cell, x:x + cell] = cl
    write_png(path, sheet)


# --------------------------------------------------------------------- README

def regen_command(args, out):
    parts = ["docker compose -f docker/compose.yaml run --rm matup python", SCRIPT]
    if os.path.normpath(args.root) != DEFAULT_ROOT:
        parts += ["--root", args.root]
    for s in args.arm:
        parts += ["--arm", s]
    parts += ["--out", out]
    if args.material:
        parts += ["--material", args.material]
    if args.run:
        parts += ["--run", args.run]
    if args.cell != DEFAULT_CELL:
        parts += ["--cell", str(args.cell)]
    if args.ov_cell != DEFAULT_OV_CELL:
        parts += ["--ov-cell", str(args.ov_cell)]
    if args.chunk != DEFAULT_CHUNK:
        parts += ["--chunk", str(args.chunk)]
    return " ".join(parts)


def manifest_cell(per, field):
    vals = sorted({str(v.get(field)) for v in per.values() if v.get(field) is not None})
    if not vals:
        return "-"
    if len(vals) == 1:
        return "`%s`" % vals[0]
    return "VARIES: " + ", ".join("`%s`" % v for v in vals)


def write_readme(path, args, out, cols, names, manifests, infos, lights, files):
    graz = lights["sheet"]["grazing"]
    deg = lambda j: math.degrees(math.asin(lights["uz_held"][j]))  # noqa: E731
    L = ["# DeepTextures arms only - maps and renders of the predicted SVBRDF",
         "",
         "Generated %s by `scripts/compare/deeptex_dt_montage.py`. A DeepTextures-only view of "
         "the render-loss study in `..`: every row is a DeepTextures checkpoint or GT. There "
         "are NO LR / bilinear / MatUp rows here - those baselines are in `../<material>.png` "
         "and `../render_<material>.png`." % datetime.date.today().isoformat(),
         "",
         "## Rows (top to bottom, the `--arm` order, GT last)",
         "",
         "| # | row label | tag | arm folder | checkpoint prefix (sr_manifest.json) | step |",
         "|---|---|---|---|---|---|"]
    for i, c in enumerate(cols, 1):
        per = manifests.get(c["key"], {})
        L.append("| %d | %s | `%s` | `%s` | %s | %s |" % (
            i, names[c["key"]], c["tag"], c["dir"], manifest_cell(per, "checkpoint"),
            manifest_cell(per, "step").replace("`", "")))
    L += ["| %d | GT | - | `%s/<material>/GT` | - | - |" % (len(cols) + 1, args.root),
          "",
          "Each row is ONE checkpoint. This recipe's global colour swings from checkpoint to "
          "checkpoint (DEEPTEX-DEVIATIONS 0.10), so judge an arm across its 30k / 35k / 40k rows, "
          "not from one cell. Thin grey lines separate the arm groups and GT.",
          "",
          "## Files",
          "",
          "| file | rows | columns | MB |",
          "|---|---|---|---|"]
    for f in files:
        base = os.path.basename(f)
        if base.startswith("dt_overview_"):
            key = base[len("dt_overview_"):-4]
            what = ("render under EVAL_LIGHT of the predicted SVBRDF" if key == "render"
                    else "the stored `%s` map" % key)
            L.append("| `%s` | materials | the arms + GT, %s | %.1f |" % (base, what, mb(f)))
        else:
            L.append("| `%s` | the arms + GT | 8 (below) | %.1f |" % (base, mb(f)))
    L += ["",
          "## Columns of `dt_<material>.png`",
          "",
          "| column | content |",
          "|---|---|",
          "| baseColor, normal, roughness, metallic, height | the stored PNG of the row: the arm's "
          "`<arm folder>/<material>/<map>.png` (8-bit), GT's `GT/<map>.png` (8-bit; height 16-bit) |",
          "| render EVAL_LIGHT | MatUp's torch renderer on the row's PREDICTED SVBRDF (all five maps), "
          "EVAL_LIGHT [0.276, 0.276, 0.921], norm 1, %.0f deg elevation |" % deg(0),
          "| render held-out grazing | the same, under held-out light #%d (%.0f deg elevation): the "
          "'grazing' sheet light of svbrdf_render_eval (1.4 x fib(16) turned %.2f deg about z, "
          ">= %.1f deg from every training light direction) |" % (
              graz, deg(graz), ev.matup_lights.HELDOUT_AZIMUTH_DEG, ev.matup_lights.MIN_SEP_DEG),
          "| \\|pred - GT\\| x%g (EVAL) | mean over RGB of \\|render - GT render\\| under EVAL_LIGHT, "
          "times HEAT_GAIN = %g (FIXED for every row and material; >= %.2f saturates), inferno "
          "colour map; grey = texels whose GT normal fails training's validity rule. Blank for "
          "the GT row |" % (ev.HEAT_GAIN, ev.HEAT_GAIN, 1.0 / ev.HEAT_GAIN),
          "",
          "## How the cells are made",
          "",
          "- **Crop.** Per material, the border is what `matup_postprocess.detect_border` measures "
          "on MatUp's `_opt` maps of the run named in the sheet title (baseColor / roughness / "
          "metallic; never normal or the pre-cropped height) - the call `deeptex_arms_report.py` and "
          "`svbrdf_render_eval.py` make. Map cells and renders cover the same %s px region." % (
              "/".join(sorted({"%d^2" % i["side"] for i in infos}))),
          "- **Maps** are drawn as `deeptex_arms_report.py` draws them: `post.deeptex_column` / the "
          "GT crop, `matup_montage.to_display` (16-bit >> 8, area-downsampled). Nothing is "
          "normalised: an all-zero metallic is black, the normal is its stored OpenGL RGB encoding, "
          "grey maps are grey.",
          "- **Renders** follow `svbrdf_render_eval.py` exactly: `load_dt` / `load_gt` "
          "(`matup_render.load_set(..., opengl=True, floor roughness)` + `position3d` of the set's "
          "own height), `prepare` (crop, `get_renorm_clamped_maps`), MatUp's renderer UNGUARDED, "
          "`to_cell` / `heat_cell`. The cells equal those of `../render_<material>.png` at a smaller "
          "size. The GT row's EVAL_LIGHT render reproduces MatUp's committed `render_gt.png` "
          "(max difference per material below, tolerance %.1f LSB)." % (ev.SELF_TOL["gt"] * ev.U16),
          "",
          "## Two meanings of \"render\"",
          "",
          "- HERE (and in `../render_<material>.png`, `../overview_render_pred.png`): the PREDICTED "
          "SVBRDF - all five maps a checkpoint produced - rendered by MatUp's renderer. This is the "
          "quantity the render loss trains.",
          "- In `../<material>.png` and `../overview_render.png` (deeptex_arms_report.py): "
          "DeepTextures run as an IMAGE super-resolver on MatUp's LR render (`render_lr.png`). No "
          "predicted map goes into it. That column is NOT in this folder.",
          "",
          "## Per material",
          "",
          "| material | MatUp run (border source) | border | crop | GT-valid texels | GT self-check (LSB) |",
          "|---|---|---|---|---|---|"]
    for i in infos:
        L.append("| %s | %s | %d | %d^2 | %.4f | %.2f |" % (
            i["material"], i["run"], i["border"], i["side"], i["valid_frac"],
            i["self_check_lsb"]))
    miss = ["%s: %s" % (i["material"], ", ".join(i["missing"])) for i in infos if i["missing"]]
    fixed = ["%s %s: %d" % (i["material"], k, v) for i in infos
             for k, v in i["nz_zero_fixed"].items()]
    L += ["",
          "Missing arm outputs (drawn as n/a): %s." % ("none" if not miss else "; ".join(miss)),
          "Texels whose rendered normal had n_z == 0 exactly (given the smallest positive n_z, "
          "as svbrdf_render_eval does): %s." % ("none" if not fixed else "; ".join(fixed)),
          "",
          "## Regenerate",
          "",
          "From the repo root (the GPU is used for the renders; a few minutes):",
          "",
          "```",
          regen_command(args, out),
          "```",
          ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# ------------------------------------------------------------------------ main

def clear_stale(out, mats):
    """Remove this script's own outputs of an earlier run (these names only)."""
    names = ["dt_overview_%s.png" % k for k in OV_KEYS]
    names += ["dt_%s.png" % os.path.basename(m.rstrip("/")) for m in mats]
    # README.md only if this script wrote it: pointed at the parent collection by
    # mistake, --out would otherwise delete that folder's committed README.
    readme = os.path.join(out, "README.md")
    if os.path.isfile(readme):
        with open(readme, encoding="utf-8", errors="replace") as f:
            if "deeptex_dt_montage.py" in f.read(2000):
                names.append("README.md")
            else:
                die(2, "%s exists and was not written by this script - choose another --out"
                    % readme)
    gone = 0
    for n in names:
        p = os.path.join(out, n)
        if os.path.isfile(p):
            os.remove(p)
            gone += 1
    if gone:
        print("[dt-montage] removed %d output file(s) of an earlier run from %s" % (gone, out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--arm", action="append", default=[],
                    help="TAG=DIR of a deeptex_compare.py --out (DIR/<mat>/{five maps}.png + "
                         "sr_manifest.json); repeat per arm, rows follow this order. A _NNk "
                         "suffix must match the manifest's step")
    ap.add_argument("--out", required=True, help="absolute output folder")
    ap.add_argument("--material", default="")
    ap.add_argument("--run", default="", help="MatUp output stamp; default is the newest")
    ap.add_argument("--cell", type=int, default=DEFAULT_CELL,
                    help="per-material sheet cell size")
    ap.add_argument("--ov-cell", type=int, default=DEFAULT_OV_CELL, help="overview cell size")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                    help="held-out lights per render call (svbrdf_render_eval --chunk)")
    args = ap.parse_args()

    out = ev.check_out(args.out)
    if not args.arm:
        die(2, "give at least one --arm TAG=DIR")
    if args.cell < 64 or args.ov_cell < 48 or args.chunk < 1:
        die(2, "--cell >= 64, --ov-cell >= 48 and --chunk >= 1 please")
    cols = ev.parse_arms(args.arm)
    mats = [os.path.join(args.root, args.material)] if args.material else \
        sorted(d for d in glob.glob(os.path.join(args.root, "*"))
               if os.path.isdir(os.path.join(d, "SVBRDF")))
    if not mats or not all(os.path.isdir(os.path.join(m, "GT")) for m in mats):
        die(2, "no material with GT/ under %s" % args.root)
    # Also refuses a _NNk tag whose manifest names another step.
    manifests = ev.read_manifests(cols, mats, None)

    # Row labels: svbrdf_render_eval.display_name; a tag without a step suffix
    # gets the step its manifests agree on ("DeepTextures (matsynth_all) 40k").
    names = {}
    for c in cols:
        label = ev.display_name(c["tag"])
        steps = c.get("manifest_steps") or []
        if not c["step"] and len(steps) == 1:
            s = int(steps[0])
            label += " %dk" % (s // 1000) if s % 1000 == 0 else " step %d" % s
        names[c["key"]] = label
    names["gt"] = "GT"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Renderer.init_renderer(argparse.Namespace(light_falloff=False), device)
    lights = ev.build_lights(device)
    if "grazing" not in lights["sheet"]:
        die(2, "no grazing held-out light (svbrdf_render_eval.build_lights)")
    graz = lights["sheet"]["grazing"]
    deg = lambda j: math.degrees(math.asin(lights["uz_held"][j]))  # noqa: E731
    os.makedirs(out, exist_ok=True)
    clear_stale(out, mats)
    print("[dt-montage] device=%s materials=%d rows=%s + GT  out=%s" % (
        device, len(mats), ",".join(c["tag"] for c in cols), out))
    print("[dt-montage] lights: EVAL_LIGHT %.1f deg, grazing held-out #%d %.1f deg; "
          "heat gain x%g" % (deg(0), graz, deg(graz), ev.HEAT_GAIN))

    heads = ["baseColor", "normal", "roughness", "metallic", "height",
             ["render (pred. SVBRDF)", "EVAL_LIGHT %.0f deg" % deg(0)],
             ["render (pred. SVBRDF)", "held-out #%d grazing %.0f deg" % (graz, deg(graz))],
             ["|pred - GT| x%g" % ev.HEAT_GAIN, "under EVAL_LIGHT"]]
    notes = ["Render columns: MatUp's torch renderer applied to each row's PREDICTED SVBRDF (all "
             "five maps, raw, unguarded) - not DeepTextures image SR of the LR render. "
             "DeepTextures arms only: no LR / bilinear / MatUp rows.",
             "|pred - GT|: RGB mean under EVAL_LIGHT x %g, FIXED for every row and material "
             "(>= %.2f saturates); grey = GT-invalid texels. Maps: stored PNGs cropped by MatUp's "
             "border, not normalised; normal = stored OpenGL RGB." % (
                 ev.HEAT_GAIN, 1.0 / ev.HEAT_GAIN)]

    infos, per_key, files = [], {k: [] for k in OV_KEYS}, []
    with torch.no_grad():
        for n, d in enumerate(mats, 1):
            info, rows, ov_cells = process(d, cols, args, device, lights)
            if info is None:
                continue
            infos.append(info)
            p = os.path.join(out, "dt_%s.png" % info["material"])
            write_sheet(p, "%s  -  DeepTextures arms only    MatUp %s" % (
                info["material"], info["run"]), notes, heads, rows, names, args.cell)
            files.append(p)
            for k in OV_KEYS:
                per_key[k].append((info["material"], ov_cells[k]))
            print("  [%2d/%d] %-28s %s border=%d crop=%d self-check gt=%.2f LSB rows=%d%s "
                  "-> %s (%.1f MB)" % (
                      n, len(mats), info["material"], info["run"], info["border"],
                      info["side"], info["self_check_lsb"], len(rows),
                      " MISSING " + ",".join(info["missing"]) if info["missing"] else "",
                      os.path.basename(p), mb(p)))
            sys.stdout.flush()
    if not infos:
        die(2, "nothing drawn")

    col_keys = [c["key"] for c in cols] + ["gt"]
    bases = [c["base"] for c in cols] + ["gt"]
    for k in OV_KEYS:
        p = os.path.join(out, "dt_overview_%s.png" % k)
        title = ("Renders of the predicted SVBRDF (MatUp renderer, EVAL_LIGHT)  -  DeepTextures "
                 "arms only  -  %d materials" % len(per_key[k]) if k == "render" else
                 "DeepTextures arms only  -  %s  -  %d materials" % (k, len(per_key[k])))
        write_overview(p, title, per_key[k], col_keys, bases, names, args.ov_cell)
        files.append(p)
        print("  overview %-10s -> %s (%.1f MB)" % (k, os.path.basename(p), mb(p)))

    write_readme(os.path.join(out, "README.md"), args, out, cols, names, manifests, infos,
                 lights, files)
    total = sum(os.path.getsize(f) for f in files) / 1e6
    print("[dt-montage] %d material(s), %d PNG(s), %.1f MB -> %s" % (
        len(infos), len(files), total, out))


if __name__ == "__main__":
    main()
