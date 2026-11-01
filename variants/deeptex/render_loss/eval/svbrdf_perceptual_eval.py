#!/usr/bin/env python3
"""LPIPS and FLIP of the renders of every method's PREDICTED SVBRDF.

Runs in the exp container (texsr/exp, built ON TOP of texsr/matup: torch 2.0,
Python 3.10, cv2, lpips 0.1.4, flip-evaluator 1.6.0.1), on the same
deeptex_compare.py --out folders svbrdf_render_eval.py scores:

    cd docker/exp
    docker compose run --rm exp python /workspace/variants/deeptex/render_loss/eval/svbrdf_perceptual_eval.py \\
        --arm matsynth_all=/workspace/runs/deeptex/eval/matsynth_all \\
        --arm render_r0a_40k=/workspace/runs/deeptex/eval/render_r0a_40k \\
        --arm render_r1_40k=/workspace/runs/deeptex/eval/render_r1_40k \\
        --out /workspace/runs/deeptex/render_eval/perceptual

run_deeptex.ps1 -Stage eval-perceptual passes every (arm, step) eval-render
evaluated. --out must be absolute; one under external/ is refused (exit 2).
The matup image has lpips but not flip-evaluator, so this is exp-only.

WHY THIS EXISTS
---------------
svbrdf_render_eval.py ranks R1 (DeepTextures + render loss) above R0a on
held-out render PSNR and l2,1, while R1's EVAL_LIGHT render keeps less detail:
|grad| 60-71% of the GT render's against R0a's 85-95% (render_summary.md). PSNR
and l2,1 are mean errors, and under uncertainty the estimate that minimises a
mean error is the conditional mean - a blur. A high frequency that does not
line up with GT costs a mean error more than leaving it out, so P1 cannot tell
R1 having removed NOISE from R1 having removed DETAIL that GT has. LPIPS (deep
features, linear heads fitted to human 2AFC judgements) and FLIP (CSF-filtered
colour difference, amplified by edge and point feature differences) charge lost
texture and edges directly, so they do not reward blur the way a mean error does:
    R1 <= R0a here    the |grad| R1 lost was mostly noise;
    R1 >  R0a here    R1 traded detail GT has for P1.
bilinear (|grad| ~35%) is the blur reference: a metric that ranks it first is
not measuring what this is for.

WHAT IS RENDERED - svbrdf_render_eval.py's pipeline, IMPORTED, not copied
  columns    bilinear, matup, one dt_<TAG> per --arm, in that order; gt is the
             reference, not a row (svbrdf_render_eval COLUMNS).
  maps       load_gt / load_bilinear / load_matup / load_dt, the border
             detect_border measures on MatUp's opt maps, prepare() - the order
             of svbrdf_render_eval.process() (:704-807), cited line by line below.
  variants   raw (as delivered) and off (remove_offsets -> decode_enc ->
             prepare: each map's per-channel mean error vs GT removed in the
             stored encoding; P1's variant).
  display    render_display (MatUp's renderer, then Reinhard + gamma,
             render_utils.process_raw_render), laid out (lights, 3, H, W) and
             clipped to [0,1] as write_image_as_png clips (file_utils.py:114):
             LPIPS and FLIP's LDR mode both want [0,1]. Floats, not quantised.
  self-check the EVAL_LIGHT renders of gt / bilinear / matup must reproduce the
             committed render_gt / render_bilinear / render_opt PNGs
             (svbrdf_render_eval.self_check, exit 3), so what is scored here is
             what render_summary.md scored, in this container too.

LIGHTS (svbrdf_render_eval.build_lights; held-out index 0 is EVAL_LIGHT)
  heldout   all 17: EVAL_LIGHT + 1.4 x fib(16) turned 43.25 deg about z, none
            within 2.5 deg of a training direction.
  sheet     EVAL_LIGHT + the grazing / mid / high held-out light the
            render_<mat>.png sheets show (build_lights :1199-1206), so a FLIP
            number can be checked against the picture.
  --lpips-lights (default heldout), --flip-lights (default sheet). The union is
  rendered --chunk lights at a time; per column only one chunk of renders is
  alive, against a cached GT stack of the union (<= 17 x 496^2 x 3 float32).

GT-INVALID TEXELS. svbrdf_render_eval scores only texels whose GT normal passes
training's rule (gt_valid_mask; 0844_syntetic_foam has back-facing GT normals).
LPIPS and FLIP filter over neighbourhoods, so a hole cannot be left out: at
those texels the prediction's render is REPLACED by the GT render before either
metric, i.e. zero difference there. Every column gets the same replacement on a
material, so columns stay comparable; masked_frac is reported per material.
baseColor is not masked: the rule is about the GT NORMAL, and an albedo map does
not depend on it.

METRICS per (material, column, variant); lower is better for every one
  render_lpips        mean over --lpips-lights of LPIPS-alex v0.1 between the
                      prediction's and GT's display render
  render_flip         mean over --flip-lights of mean FLIP (LDR, default
                      parameters; the parameters FLIP reports are in the JSON)
  render_lpips_eval,  EVAL_LIGHT alone - the light render_summary.md's |grad|
  render_flip_eval    %GT is measured under
  basecolor_lpips     LPIPS-alex of the stored sRGB baseColor map, border-cropped;
                      for "off" the offset-removed one
  *_per_light         every light's value, in perceptual_report.json / the CSV
The two calls follow experiments/common/metrics.py (:34-56 LPIPS, :123-140 FLIP);
they are restated, not imported - scripts/ does not depend on experiments/.

METRIC SELF-CHECK (exit 3, on the first material scored). LPIPS(gt, gt) and
FLIP(gt, gt) <= 1e-6 under every scored light, baseColor LPIPS(gt, gt) too
(experiments/gates/gate_protocol.py:121 holds the same identity to the same
bound): fails a metric that is not zero on identical inputs. Both > 0.05 for
the GT render against its negative (1 - GT) under EVAL_LIGHT: fails a constant
or zero return. Neither half catches a wrong LPIPS input range - a missing or
doubled x * 2 - 1 still gives 0 on identical images and a large value on
opposite ones - so lpips_batch(1 - GT, GT) must also match, to 1e-6, lpips's
own [0,1] entry point on the same pair (LPIPS.forward(..., normalize=True),
lpips 0.1.4, which does 2 * x - 1 itself). FLIP's argument order (reference
first, metrics.py:133) is not pinned: both images go through the same filters
and LDR-FLIP takes absolute differences, so a swap would barely move a number.

--threads N caps torch's (renderer, LPIPS) and OpenCV's CPU threads; without it
torch takes one per core the Docker VM exposes. run_deeptex.ps1 passes it with
-EvalDevice cpu, which is how this runs beside a training job's input pipeline.

OUTPUTS (all in --out): perceptual_report.json, perceptual_long.csv,
perceptual_summary.md (means per column and variant, non-metal / metal split,
head-to-head counts over svbrdf_render_eval.head_to_head_pairs, the per-step
table). Those three of an earlier run are removed first; a missing or
unreadable map is exit 2, a non-finite render exit 4 (svbrdf_render_eval).
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time

# Before anything from external/ is imported: Python 3.10 must not leave
# cpython-310.pyc files inside the submodule (svbrdf_render_eval.py:143-145).
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4)))
from variants.common import paths  # noqa: E402

# Its import puts MatUp on sys.path (paths.use_matup()) and binds mr, post,
# data_utils and Renderer; the names below come from there.
from variants.deeptex.render_loss.eval import svbrdf_render_eval as sre  # noqa: E402

mr, post, data_utils, Renderer = sre.mr, sre.post, sre.data_utils, sre.Renderer

# ------------------------------------------------------------------- constants

VARIANTS = sre.VARIANTS                       # ("off", "raw")
MAIN = ("render_lpips", "render_flip", "basecolor_lpips")
SCALARS = ("render_lpips", "render_flip", "render_lpips_eval", "render_flip_eval",
           "basecolor_lpips")
TITLE = {"render_lpips": "render LPIPS", "render_flip": "render FLIP",
         "render_lpips_eval": "EVAL_LIGHT LPIPS", "render_flip_eval": "EVAL_LIGHT FLIP",
         "basecolor_lpips": "baseColor LPIPS"}
IDENTITY_TOL = 1e-6        # experiments/gates/gate_protocol.py:121
RESPONSE_MIN = 0.05        # GT vs 1 - GT: far above this for any textured render
OUTPUTS = ("perceptual_report.json", "perceptual_long.csv", "perceptual_summary.md")

# Per-step tables: (label, metric, variant, subset), steps_tables' layout (:907-931).
STEP_TABLES = (("render LPIPS (off)", "render_lpips", "off", "all"),
               ("render LPIPS (raw)", "render_lpips", "raw", "all"),
               ("render FLIP (off)", "render_flip", "off", "all"),
               ("render FLIP (raw)", "render_flip", "raw", "all"),
               ("baseColor LPIPS (off)", "basecolor_lpips", "off", "all"),
               ("baseColor LPIPS (raw)", "basecolor_lpips", "raw", "all"),
               ("render LPIPS (off), non-metal", "render_lpips", "off", "nonmetal"),
               ("render FLIP (off), non-metal", "render_flip", "off", "nonmetal"))

# Head-to-head keys "<metric>_<variant>", in table order.
H2H = [(m, v) for m in MAIN for v in VARIANTS]


# --------------------------------------------------------------------- metrics

class Perceptual:
    """LPIPS-alex and FLIP as experiments/common/metrics.py calls them."""

    def __init__(self, device):
        try:
            import flip_evaluator
            import lpips
        except ImportError as e:
            sre.die(2, "%s - run this in the exp container (docker/exp/compose.yaml, "
                       "service exp): the matup image has lpips but not flip-evaluator" % e)
        # metrics.py:39. AlexNet's ImageNet weights come from TORCH_HOME
        # (/expdata/cache/torch in the exp image); the linear heads ship with lpips.
        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.flip = flip_evaluator
        self.flip_params = None
        self.versions = {"lpips": getattr(lpips, "__version__", None),
                         "flip_evaluator": getattr(flip_evaluator, "__version__", None)}

    def lpips_batch(self, pred, gt):
        """(n, 3, H, W) sRGB/display images in [0,1] on the net's device -> (n,)
        float64. LPIPS takes [-1,1] (metrics.py:53-54: x * 2 - 1)."""
        d = self.net(pred * 2.0 - 1.0, gt * 2.0 - 1.0)
        return d.reshape(-1).double().cpu().numpy()

    def lpips_lib01(self, pred, gt):
        """lpips_batch's inputs through lpips's OWN [0,1] path: lpips 0.1.4
        LPIPS.forward(in0, in1, retPerLayer=False, normalize=False) does
        2 * x - 1 itself when normalize=True. metric_check alone calls it, to pin
        lpips_batch's shift against the library rather than against metrics.py."""
        try:
            d = self.net(pred, gt, normalize=True)
        except TypeError as e:
            sre.die(3, "lpips %s: LPIPS.forward has no normalize= flag (%s), so the input-range "
                       "self-check cannot run" % (self.versions["lpips"], e))
        return d.reshape(-1).double().cpu().numpy()

    def flip_mean(self, ref, test):
        """Mean FLIP of two HxWx3 float32 sRGB images in [0,1], REFERENCE first:
        flip.evaluate(reference, test, "LDR", applyMagma=False)[1] (metrics.py:131-133)."""
        res = self.flip.evaluate(np.ascontiguousarray(ref, dtype=np.float32),
                                 np.ascontiguousarray(test, dtype=np.float32),
                                 "LDR", applyMagma=False)
        if self.flip_params is None and len(res) > 2:
            p = res[2]
            self.flip_params = ({str(k): v if isinstance(v, (bool, int, float, str)) else str(v)
                                 for k, v in p.items()} if isinstance(p, dict) else str(p))
        return float(res[1])


def map_lpips(metric, pred, gt, device, where):
    """A stored sRGB map (HxWx3, [0,1]) -> LPIPS vs GT (metrics.lpips_srgb, :47-56)."""
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[2] != 3:
        sre.die(2, "%s: baseColor is %s, GT %s" % (where, pred.shape, gt.shape))

    def t(a):
        a = np.ascontiguousarray(np.clip(a, 0.0, 1.0), dtype=np.float32)
        return torch.from_numpy(a).permute(2, 0, 1)[None].to(device)
    return float(metric.lpips_batch(t(pred), t(gt))[0])


# ---------------------------------------------------------------- lights, renders

def light_sets(lights, args):
    """Held-out indices per metric and the union that gets rendered."""
    named = {"heldout": list(range(len(lights["held_np"]))),
             "sheet": [0] + sorted(int(j) for j in lights["sheet"].values())}
    lp, fl = named[args.lpips_lights], named[args.flip_lights]
    union = sorted(set(lp) | set(fl))
    return {"lpips": lp, "flip": fl, "union": union,
            "pos": {j: i for i, j in enumerate(union)},
            "union_t": lights["held"][union]}


def render_images(rmaps, lights_t, lo, hi, side, where):
    """(hi - lo, 3, side, side) display images in [0,1] for lights lo:hi.

    render_display returns (N, 3, l) with N the texels in row-major order
    (render_utils.reshape_maps_input_render), the layout eval_image reshapes to
    (h, w, 3) - so (l, 3, N) -> (l, 3, h, w) here. The clip is write_image_as_png's
    (file_utils.py:114); Reinhard + gamma already land in [0,1] for any
    non-negative radiance.
    """
    d = sre.render_display(rmaps, lights_t[lo:hi], where)
    return d.permute(2, 1, 0).reshape(hi - lo, 3, side, side).clamp(0.0, 1.0)


def gt_stack(gt_rmaps, sets, chunk, side, where):
    """The GT renders of the union lights, cached once per material."""
    return torch.cat([render_images(gt_rmaps, sets["union_t"], lo, hi, side, where)
                      for lo, hi in sre.spans(len(sets["union"]), chunk)])


def hwc(t):
    return t.permute(1, 2, 0).cpu().numpy()


def mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def score_column(rmaps, ctx, metric, chunk, where):
    """LPIPS / FLIP of one column's renders, one light chunk alive at a time.

    GT-invalid texels take the GT render first (module docstring). LPIPS runs
    on the chunk's selected lights as one batch; FLIP one light at a time on
    numpy, as metrics.render_metrics does.
    """
    sets, gt = ctx["sets"], ctx["gt"]
    lp_set, fl_set = set(sets["lpips"]), set(sets["flip"])
    lp, fl = {}, {}
    for lo, hi in sre.spans(len(sets["union"]), chunk):
        img = render_images(rmaps, sets["union_t"], lo, hi, ctx["side"], where)
        g = gt[lo:hi]
        img = torch.where(ctx["invalid"], g, img)
        held = sets["union"][lo:hi]
        sel = [i for i, j in enumerate(held) if j in lp_set]
        if sel:
            for i, v in zip(sel, metric.lpips_batch(img[sel], g[sel])):
                lp[held[i]] = float(v)
        for i, j in enumerate(held):
            if j in fl_set:
                fl[j] = metric.flip_mean(ctx["gt_np"][j], hwc(img[i]))
        del img
    lp_l = [lp[j] for j in sets["lpips"]]
    fl_l = [fl[j] for j in sets["flip"]]
    return {"render_lpips": mean(lp_l), "render_flip": mean(fl_l),
            "render_lpips_eval": lp.get(0), "render_flip_eval": fl.get(0),
            "render_lpips_per_light": lp_l, "render_flip_per_light": fl_l}


# ---------------------------------------------------------------- self-check

def metric_check(ctx, metric, gt_bc, device, chunk, name):
    """LPIPS / FLIP of GT against itself must be 0, against 1 - GT clearly not,
    and lpips_batch must agree with lpips's own [0,1] path (module docstring)."""
    sets, gt, pos = ctx["sets"], ctx["gt"], ctx["sets"]["pos"]
    li = [pos[j] for j in sets["lpips"]]
    lp_id = []
    for lo, hi in sre.spans(len(li), chunk):
        x = gt[li[lo:hi]]
        lp_id += [float(v) for v in metric.lpips_batch(x, x)]
    fl_id = [metric.flip_mean(ctx["gt_np"][j], ctx["gt_np"][j]) for j in sets["flip"]]
    bc_id = map_lpips(metric, gt_bc, gt_bc, device, name + " baseColor")
    e = gt[pos[0]:pos[0] + 1]
    lp_neg = float(metric.lpips_batch(1.0 - e, e)[0])
    # The one pair whose inputs differ: identical inputs give 0 in any range.
    lp_lib = float(metric.lpips_lib01(1.0 - e, e)[0])
    fl_neg = metric.flip_mean(ctx["gt_np"][0], 1.0 - ctx["gt_np"][0])
    rec = {"material": name, "lpips_gt_gt_max": max(lp_id), "flip_gt_gt_max": max(fl_id),
           "basecolor_lpips_gt_gt": bc_id, "lpips_gt_vs_negative_eval": lp_neg,
           "lpips_gt_vs_negative_eval_lib01": lp_lib,
           "lpips_range_diff_eval": abs(lp_neg - lp_lib),
           "flip_gt_vs_negative_eval": fl_neg, "identity_tol": IDENTITY_TOL,
           "response_min": RESPONSE_MIN}
    bad = []
    for k in ("lpips_gt_gt_max", "flip_gt_gt_max", "basecolor_lpips_gt_gt",
              "lpips_range_diff_eval"):
        if not rec[k] <= IDENTITY_TOL:
            bad.append("%s = %.3g > %g" % (k, rec[k], IDENTITY_TOL))
    for k in ("lpips_gt_vs_negative_eval", "flip_gt_vs_negative_eval"):
        if not rec[k] > RESPONSE_MIN:
            bad.append("%s = %.3g <= %g" % (k, rec[k], RESPONSE_MIN))
    rec["pass"] = not bad
    print("  [check] %s: LPIPS(gt,gt) max %.2e, FLIP(gt,gt) max %.2e, baseColor %.2e; "
          "vs 1-GT (EVAL_LIGHT) LPIPS %.3f FLIP %.3f; LPIPS vs normalize=True %.2e" % (
              name, rec["lpips_gt_gt_max"], rec["flip_gt_gt_max"], bc_id, lp_neg, fl_neg,
              rec["lpips_range_diff_eval"]))
    if bad:
        sre.die(3, "metric self-check FAILED on %s: %s. The LPIPS / FLIP call does not "
                   "behave as metrics.py assumes; every number would be meaningless."
                % (name, "; ".join(bad)))
    return rec


# --------------------------------------------------------------- per material

def measure_border(run_dir):
    """svbrdf_render_eval.process :710-718 (deeptex_arms_report.py:166-177):
    MatUp's zero rim, never measured on normal or the pre-cropped height."""
    opts = {}
    for m, (_, opt_name) in post.MAPS.items():
        p = os.path.join(run_dir, opt_name)
        if os.path.exists(p):
            opts[m] = post.read(p)
    return post.detect_border(
        [a for m, a in opts.items() if m != "normal" and m not in post.PRECROPPED])


def load_column(col, mat_dir, run_dir, full, name):
    """svbrdf_render_eval.process :766-784. None when a DT arm lacks a map."""
    if col["kind"] == "bilinear":
        maps, enc, floor = sre.load_bilinear(mat_dir, full)
    elif col["kind"] == "matup":
        maps, enc, floor = sre.load_matup(mat_dir, run_dir)
    else:
        d = os.path.join(col["dir"], name)
        need = [os.path.join(d, m + ".png") for m in sre.MAP_ORDER]
        lost = [p for p in need if not os.path.exists(p)]
        if lost:
            print("  ! %s: arm %s has no %s" % (name, col["tag"], ", ".join(
                os.path.basename(p) for p in lost)))
            return None
        sre.require_readable(need, "%s / %s" % (name, col["key"]))
        maps, enc, floor = sre.load_dt(col["dir"], name)
    if maps[0].shape[0] != full or any(t.shape[:2] != maps[0].shape[:2] for t in maps):
        sre.die(2, "%s / %s: map sizes %s, GT is %d" % (
            name, col["key"], [tuple(t.shape) for t in maps], full))
    return maps, enc, floor


def process(mat_dir, cols, args, device, lights, sets, metric, check):
    t0 = time.time()
    name = os.path.basename(mat_dir.rstrip("/"))
    run_dir = sre.newest_run(mat_dir, name, args.run)
    if not run_dir:
        print("  %-28s SKIP (no MatUp output)" % name)
        return None
    # :704-708 - gt, bilinear and matup are always columns; a hole is fatal.
    sre.require_readable([os.path.join(mat_dir, sub, m + ".png")
                          for sub in ("GT", "SVBRDF") for m in sre.MAP_ORDER]
                         + [os.path.join(run_dir, m + "_opt.png")
                            for m in sre.MAP_ORDER if m != "height"], name)

    border, sides = measure_border(run_dir)
    gt_maps, gt_enc_full, _ = sre.load_gt(mat_dir)                                    # :720
    full = gt_maps[0].shape[0]
    side = full - 2 * border
    pos2d = mr.crop(data_utils.compute_2D_position_grid(full).float(), border)         # :723
    mask_hw = sre.gt_valid_mask(os.path.join(mat_dir, "GT"), border)                   # :724
    n_valid = int(mask_hw.sum())
    if n_valid == 0:
        sre.die(2, "%s: no GT texel passes the validity rule" % name)
    metal = bool(sre.read_raw(os.path.join(mat_dir, "GT", "metallic.png"), False).max() > 0)
    valid_frac = n_valid / float(side * side)

    rep = {"material": name, "run": os.path.basename(run_dir), "border": border,
           "border_sides": list(sides), "size": side, "metal": metal, "n_valid": n_valid,
           "valid_frac": valid_frac, "masked_frac": 1.0 - valid_frac,
           "self_check": {}, "columns": {}, "missing": []}

    gt_rmaps, _ = sre.prepare(gt_maps, border, device)                                  # :735
    gt_enc = sre.crop_enc(gt_enc_full, border)                                          # :736
    rep["self_check"]["gt"] = sre.self_check(                                           # :737-738
        "gt", sre.eval_image(gt_rmaps, lights["eval"], "%s gt eval" % name), run_dir, border, full)
    gt = gt_stack(gt_rmaps, sets, args.chunk, side, "%s gt" % name)
    del gt_rmaps, gt_maps
    ctx = {"sets": sets, "side": side, "gt": gt,
           "gt_np": {j: hwc(gt[sets["pos"][j]]) for j in set(sets["flip"]) | {0}},
           "invalid": torch.from_numpy(~mask_hw)[None, None].to(device)}
    if check:
        rep["metric_check"] = metric_check(ctx, metric, gt_enc["baseColor"], device,
                                           args.chunk, name)

    for col in cols:
        key = col["key"]
        got = load_column(col, mat_dir, run_dir, full, name)
        if got is None:
            rep["missing"].append(key)
            continue
        maps, enc, floor = got
        where = "%s %s" % (name, key)
        enc_c = sre.crop_enc(enc, border)
        entry = {}

        raw, fix = sre.prepare(maps, border, device)                                    # :788
        if col["kind"] in sre.SELF_TOL:                                                  # :792-793
            rep["self_check"][col["kind"]] = sre.self_check(
                col["kind"], sre.eval_image(raw, lights["eval"], where + " eval"),
                run_dir, border, full)
        m = score_column(raw, ctx, metric, args.chunk, where + " raw")
        m["basecolor_lpips"] = map_lpips(metric, enc_c["baseColor"], gt_enc["baseColor"],
                                         device, where)
        m["nz_zero_fixed"], m["zero_normals"] = fix, 0
        entry["raw"] = m
        del raw, maps

        enc_off, offsets, _ = sre.remove_offsets(enc_c, gt_enc, where)                  # :798
        off_maps, zero_n = sre.decode_enc(enc_off, floor, pos2d)                         # :799
        off, fix = sre.prepare(off_maps, 0, device)                                     # :800
        m = score_column(off, ctx, metric, args.chunk, where + " off")
        m["basecolor_lpips"] = map_lpips(metric, enc_off["baseColor"], gt_enc["baseColor"],
                                         device, where + " off")
        m["nz_zero_fixed"], m["zero_normals"] = fix, zero_n
        m["offset_255"] = offsets
        entry["off"] = m
        del off, off_maps
        rep["columns"][key] = entry

    sc = rep["self_check"]
    print("  %-28s %s border=%d masked=%.4f%s self-check(LSB) %s  %.0fs" % (
        name, rep["run"], border, rep["masked_frac"], " metal" if metal else "",
        " ".join("%s=%.1f" % (c, sc[c]["max_abs"] * sre.U16)
                 for c in ("gt", "bilinear", "matup") if c in sc), time.time() - t0))
    for metric_key, label in (("render_lpips", "render LPIPS   "),
                              ("render_flip", "render FLIP    "),
                              ("basecolor_lpips", "baseColor LPIPS")):
        print("      %s off/raw: %s" % (label, "  ".join(
            "%s %s/%s" % (c["short"], sre.fmt(rep["columns"][c["key"]]["off"][metric_key], "%.4f"),
                          sre.fmt(rep["columns"][c["key"]]["raw"][metric_key], "%.4f"))
            for c in cols if c["key"] in rep["columns"])))
    return rep


# ----------------------------------------------------------------- aggregation

def common_materials(reports, keys):
    """svbrdf_render_eval.common_materials (:832-837) on these metrics: every
    mean in a table is over the same set."""
    return [r for r in reports
            if all(k in r["columns"] and all(r["columns"][k][v].get(m) is not None
                                             for v in VARIANTS for m in MAIN)
                   for k in keys)]


def head_to_head(reports, a, b):
    """svbrdf_render_eval.head_to_head (:874-904) for LPIPS / FLIP: materials where
    a is strictly LOWER (better) than b, per metric and variant, the same for the
    non-metal ones, and the mean paired difference a - b (negative = a better)."""
    keys = ["%s_%s" % mv for mv in H2H]
    wins, wins_nm = dict.fromkeys(keys, 0), dict.fromkeys(keys, 0)
    diffs = {k: [] for k in keys}
    out = {"n": 0, "n_nonmetal": 0}
    for r in reports:
        if a not in r["columns"] or b not in r["columns"]:
            continue
        ea, eb = r["columns"][a], r["columns"][b]
        vals = {"%s_%s" % (m, v): (ea[v].get(m), eb[v].get(m)) for m, v in H2H}
        if any(x is None or y is None for x, y in vals.values()):
            continue
        out["n"] += 1
        out["n_nonmetal"] += not r["metal"]
        for k, (x, y) in vals.items():
            wins[k] += x < y
            wins_nm[k] += (x < y) and not r["metal"]
            diffs[k].append(x - y)
    out["n_nonmetal"] = int(out["n_nonmetal"])
    out["wins"] = {k: int(v) for k, v in wins.items()}
    out["wins_nonmetal"] = {k: int(v) for k, v in wins_nm.items()}
    out["mean_diff"] = {k: mean(d) for k, d in diffs.items()}
    return out


def steps_tables(common, cols):
    """svbrdf_render_eval.steps_tables (:907-931) for these metrics: per base arm
    with step-suffixed tags, the mean at each step, over steps, and the range."""
    dts = [c for c in cols if c["kind"] == "dt" and c["step"]]
    bases = []
    for c in dts:
        if c["base"] not in bases:
            bases.append(c["base"])
    steps = sorted({c["step"] for c in dts}, key=sre.step_key)
    key_of = {(c["base"], c["step"]): c["key"] for c in dts}
    out = {}
    for label, metric, variant, subset in STEP_TABLES:
        rs = common if subset == "all" else [r for r in common if not r["metal"]]
        rows = {}
        for base in bases:
            vals = {s: sre.mean_of(rs, key_of[(base, s)], variant, metric)
                    for s in steps if (base, s) in key_of}
            got = [v for v in vals.values() if v is not None]
            rows[base] = {"steps": vals,
                          "mean": float(np.mean(got)) if got else None,
                          "range": float(max(got) - min(got)) if len(got) > 1 else None}
        out[label] = {"steps": steps, "rows": rows, "n": len(rs)}
    return out


def elev_deg(lights, j):
    return math.degrees(math.asin(float(lights["uz_held"][j])))


def light_desc(lights, idx):
    return ", ".join("%s %.0f deg" % ("EVAL_LIGHT" if j == 0 else "#%d" % j, elev_deg(lights, j))
                     for j in idx)


def write_summary(path, reports, cols, names, lights, sets, args, h2h, steps, check):
    keys = [c["key"] for c in cols]
    common = common_materials(reports, keys)
    nonmetal = [r for r in common if not r["metal"]]
    metal = [r for r in common if r["metal"]]
    L = ["# Perceptual metrics on renders of the predicted SVBRDF: %d MatSynth evaluation "
         "materials" % len(reports),
         "",
         "LPIPS and FLIP between the MatUp-renderer render of each column's predicted maps and",
         "the render of the GT maps - the renders svbrdf_render_eval.py scores with PSNR / l2,1",
         "(render_summary.md). Mean errors reward blur; these two charge lost texture and edges,",
         "so they separate \"R1 removed noise\" (R1 <= R0a here) from \"R1 removed detail GT has\"",
         "(R1 > R0a here). bilinear is the blur reference. **Lower is better everywhere.**",
         "",
         "- **render LPIPS**: LPIPS-alex v0.1 per light, mean over the `%s` set (%d lights: %s)."
         % (args.lpips_lights, len(sets["lpips"]), light_desc(lights, sets["lpips"])),
         "- **render FLIP**: mean FLIP (LDR, default parameters) per light, mean over the `%s` "
         "set (%d lights: %s)." % (args.flip_lights, len(sets["flip"]),
                                  light_desc(lights, sets["flip"])),
         "- **EVAL_LIGHT LPIPS / FLIP**: EVAL_LIGHT alone - the light render_summary.md's",
         "  `|grad| %GT` is measured under.",
         "- **baseColor LPIPS**: LPIPS-alex of the stored sRGB baseColor map (no lighting).",
         "- Renders: display space (Reinhard + gamma, clipped to [0,1]), float, cropped by",
         "  MatUp's border exactly as in render_summary.md.",
         "- **off** = offset-removed maps (P1's variant: each map's per-channel mean error vs",
         "  GT removed in the stored encoding); **raw** = the maps as delivered.",
         "- GT-invalid texels (GT normal fails training's rule) take the GT render in every",
         "  prediction before LPIPS / FLIP, the same for every column (masked fraction per",
         "  material at the end). baseColor is not masked.",
         "- `n`: materials every column was scored on; every mean is over that set",
         "  (%d non-metal = GT metallic all zero, %d metal)." % (len(nonmetal), len(metal)),
         "",
         "Columns: " + ", ".join("`%s` = %s" % (c["key"], names[c["key"]]) + (
             " (%s)" % c["dir"] if c.get("dir") else "") for c in cols) + ".", ""]

    tables = {}
    for v in VARIANTS:
        L += ["## %s - %s" % (v, sre.VARIANT_TITLE[v]), "",
              "| column | n | " + " | ".join(TITLE[s] for s in SCALARS) + " |",
              "|---|---|" + "---|" * len(SCALARS)]
        tables[v] = {}
        for c in cols:
            k = c["key"]
            mv = {s: sre.mean_of(common, k, v, s) for s in SCALARS}
            tables[v][k] = mv
            L.append("| %s | %d | %s |" % (names[k], len(common), " | ".join(
                sre.fmt(mv[s], "%.4f") for s in SCALARS)))
        L += ["", "Non-metal / metal split:", "",
              "| column | non-metal n | " + " | ".join(TITLE[s] for s in MAIN)
              + " | metal n | " + " | ".join(TITLE[s] for s in MAIN) + " |",
              "|---|" + "---|" * (2 + 2 * len(MAIN))]
        for c in cols:
            k = c["key"]
            cells = []
            for rs in (nonmetal, metal):
                cells += ["%d" % len(rs)] + [sre.fmt(sre.mean_of(rs, k, v, s), "%.4f")
                                             for s in MAIN]
            L.append("| %s | %s |" % (names[k], " | ".join(cells)))
        L.append("")

    L += ["## Head-to-head", "",
          "Cells `k/n`: materials where the FIRST column is strictly LOWER (better). Mean diff",
          "= first - second over the same materials; negative means the first is better.",
          "Pairs as in render_summary.md (svbrdf_render_eval.head_to_head_pairs).", "",
          "| first | second | n | " + " | ".join("%s %s" % (TITLE[m], v) for m, v in H2H)
          + " | non-metal: render LPIPS off | non-metal: render FLIP off "
            "| mean diff render LPIPS off | mean diff render FLIP off "
            "| mean diff baseColor LPIPS off |",
          "|---|---|---|" + "---|" * (len(H2H) + 5)]
    for (a, b), hh in h2h.items():
        n = hh["n"]
        if not n:
            continue
        w, wn, md = hh["wins"], hh["wins_nonmetal"], hh["mean_diff"]
        L.append("| %s | %s | %d | %s | %d/%d | %d/%d | %s | %s | %s |" % (
            names[a], names[b], n,
            " | ".join("%d/%d" % (w["%s_%s" % mv], n) for mv in H2H),
            wn["render_lpips_off"], hh["n_nonmetal"], wn["render_flip_off"], hh["n_nonmetal"],
            sre.fmt(md["render_lpips_off"], "%+.4f"), sre.fmt(md["render_flip_off"], "%+.4f"),
            sre.fmt(md["basecolor_lpips_off"], "%+.4f")))
    L.append("")

    if any(t["rows"] for t in steps.values()):
        L += ["## Per checkpoint step", "",
              "Mean over the `n` materials at each step, the mean over steps, and the",
              "within-run range (max - min over steps).", ""]
        for label, t in steps.items():
            if not t["rows"]:
                continue
            L += ["### %s (n=%d)" % (label, t["n"]), "",
                  "| arm | " + " | ".join(t["steps"]) + " | mean | range |",
                  "|---|" + "---|" * (len(t["steps"]) + 2)]
            for base, row in t["rows"].items():
                L.append("| %s | %s | %s | %s |" % (
                    sre.DISPLAY.get(base, base),
                    " | ".join(sre.fmt(row["steps"].get(s), "%.4f") for s in t["steps"]),
                    sre.fmt(row["mean"], "%.4f"), sre.fmt(row["range"], "%.4f")))
            L.append("")

    L += ["## Checks", "",
          "Metric self-check on %s: LPIPS(gt, gt) max %.2e, FLIP(gt, gt) max %.2e, baseColor "
          "LPIPS(gt, gt) %.2e (bound %g); GT vs 1 - GT under EVAL_LIGHT: LPIPS %.3f, FLIP %.3f "
          "(must exceed %g); LPIPS input range: |x * 2 - 1 here - lpips normalize=True| on that "
          "pair %.2e (bound %g)." % (check["material"], check["lpips_gt_gt_max"],
                                     check["flip_gt_gt_max"], check["basecolor_lpips_gt_gt"],
                                     IDENTITY_TOL, check["lpips_gt_vs_negative_eval"],
                                     check["flip_gt_vs_negative_eval"], RESPONSE_MIN,
                                     check["lpips_range_diff_eval"], IDENTITY_TOL), "",
          "| material | masked fraction | self-check GT (LSB) | bilinear | MatUp | missing |",
          "|---|---|---|---|---|---|"]
    for r in reports:
        sc = r["self_check"]
        L.append("| %s | %.4f | %s | %s |" % (
            r["material"], r["masked_frac"], " | ".join(
                sre.fmt(sc[c]["max_abs"] * sre.U16 if c in sc else None, "%.2f")
                for c in ("gt", "bilinear", "matup")),
            ", ".join(r["missing"]) or "-"))
    L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return tables, len(common)


def write_csv(path, reports, cols, names):
    fields = (["material", "metal", "column", "display", "tag", "step", "variant",
               "n_valid", "valid_frac", "masked_frac"] + list(SCALARS)
              + ["nz_zero_fixed", "zero_normals", "render_lpips_per_light",
                 "render_flip_per_light"])
    meta = {c["key"]: c for c in cols}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in reports:
            for k, e in r["columns"].items():
                c = meta[k]
                for v in VARIANTS:
                    row = {"material": r["material"], "metal": r["metal"], "column": k,
                           "display": names[k], "tag": c.get("tag") or "",
                           "step": c.get("step") or "", "variant": v,
                           "n_valid": r["n_valid"], "valid_frac": r["valid_frac"],
                           "masked_frac": r["masked_frac"],
                           "nz_zero_fixed": e[v].get("nz_zero_fixed"),
                           "zero_normals": e[v].get("zero_normals")}
                    row.update({s: e[v].get(s) for s in SCALARS})
                    for s in ("render_lpips_per_light", "render_flip_per_light"):
                        row[s] = ";".join("%.6f" % x for x in e[v].get(s, []))
                    w.writerow(row)


# ------------------------------------------------------------------------ main

def clear_stale(out):
    """svbrdf_render_eval.clear_stale for this script's three names only: a run
    that dies never leaves an older summary looking current (run_deeptex.ps1
    collects them by name)."""
    gone = 0
    for n in OUTPUTS:
        p = os.path.join(out, n)
        if os.path.isfile(p):
            os.remove(p)
            gone += 1
    if gone:
        print("[perceptual] removed %d output file(s) of an earlier run from %s" % (gone, out))


def pick_device(spec):
    """--device, or svbrdf_render_eval's rule (:1291). An explicit cpu never
    touches CUDA, so no context lands on a GPU a training run is using."""
    if not spec:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(spec)
    except RuntimeError as e:
        sre.die(2, "--device %r: %s" % (spec, e))
    if device.type == "cuda" and not torch.cuda.is_available():
        sre.die(2, "--device %s but CUDA is not available in this container" % spec)
    return device


def cap_threads(n):
    """--threads N, before the renderer, LPIPS or FLIP do any work. torch's
    intra-op pool (MatUp's renderer, LPIPS) and OpenCV's (svbrdf_render_eval's
    imports, map reading) both default to every core the Docker VM exposes,
    which a training run's input pipeline is using. OMP_NUM_THREADS is set too,
    for an OpenMP runtime first loaded after this point (flip_evaluator is
    imported in Perceptual); torch's own has already read the environment."""
    os.environ["OMP_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    sre.cv2.setNumThreads(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/runs/matsynth")
    ap.add_argument("--arm", action="append", default=[],
                    help="TAG=DIR of a deeptex_compare.py --out (DIR/<mat>/{five maps}.png "
                         "+ sr_manifest.json); repeat per arm. Same tags as "
                         "svbrdf_render_eval.py; a _NNk suffix must match the manifest's step")
    ap.add_argument("--out", required=True, help="absolute output folder")
    ap.add_argument("--material", default="")
    ap.add_argument("--run", default="", help="MatUp output stamp; default is the newest")
    ap.add_argument("--chunk", type=int, default=10,
                    help="lights rendered (and LPIPS-batched) per call")
    ap.add_argument("--expect-step", type=int, default=None,
                    help="every arm's sr_manifest.json step must equal this")
    ap.add_argument("--flip-lights", choices=("sheet", "heldout"), default="sheet",
                    help="sheet = EVAL_LIGHT + the grazing/mid/high sheet light; heldout = all 17")
    ap.add_argument("--lpips-lights", choices=("heldout", "sheet"), default="heldout")
    ap.add_argument("--device", default="",
                    help="cpu or cuda[:N]; default cuda:0 when available. cpu leaves the GPU "
                         "alone (for running beside a training job)")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads for torch and OpenCV; 0 = their default (every core). "
                         "Cap it with --device cpu beside a training job")
    args = ap.parse_args()

    out = sre.check_out(args.out)
    if args.chunk < 1:
        sre.die(2, "--chunk must be >= 1")
    if args.threads < 0:
        sre.die(2, "--threads must be >= 0")
    if args.threads:
        cap_threads(args.threads)
    arm_cols = sre.parse_arms(args.arm)
    cols = ([{"key": "bilinear", "kind": "bilinear", "short": "bilinear"},
             {"key": "matup", "kind": "matup", "short": "matup"}] + arm_cols)
    names = {c["key"]: sre.FIXED_NAMES.get(c["key"]) or sre.display_name(c["tag"]) for c in cols}

    # svbrdf_render_eval.main :1284-1288.
    mats = [os.path.join(args.root, args.material)] if args.material else \
        sorted(d for d in glob.glob(os.path.join(args.root, "*"))
               if os.path.isdir(os.path.join(d, "SVBRDF")))
    if not mats or not all(os.path.isdir(os.path.join(m, "GT")) for m in mats):
        sre.die(2, "no material with GT/ under %s" % args.root)
    manifests = sre.read_manifests(cols, mats, args.expect_step)

    device = pick_device(args.device)
    # LPIPS's convolutions in full float32 on Ampere (cuDNN would use TF32), so a
    # --device cpu run and a GPU run agree to float32 rounding. The renderer has
    # no convolution; matmul TF32 is already off by default in torch 2.0.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # matup_render.py:215-217: mandatory, light falloff off (svbrdf_render_eval :1292-1293).
    Renderer.init_renderer(argparse.Namespace(light_falloff=False), device)
    lights = sre.build_lights(device)
    sets = light_sets(lights, args)
    metric = Perceptual(device)
    os.makedirs(out, exist_ok=True)
    clear_stale(out)
    print("[perceptual] device=%s threads=%d materials=%d columns=%s chunk=%d out=%s" % (
        device, torch.get_num_threads(), len(mats), ",".join(c["key"] for c in cols),
        args.chunk, out))
    print("[perceptual] LPIPS lights (%s): %s" % (args.lpips_lights,
                                                  light_desc(lights, sets["lpips"])))
    print("[perceptual] FLIP lights (%s): %s" % (args.flip_lights,
                                                 light_desc(lights, sets["flip"])))
    print("[perceptual] lpips %s, flip_evaluator %s" % (
        metric.versions["lpips"], metric.versions["flip_evaluator"]))

    reports, check = [], None
    with torch.no_grad():
        for d in mats:
            rep = process(d, cols, args, device, lights, sets, metric, check is None)
            if rep is None:
                continue
            if check is None:
                check = rep["metric_check"]
            reports.append(rep)
    if not reports:
        sre.die(2, "nothing scored")

    keys = [c["key"] for c in cols]
    common = common_materials(reports, keys)
    h2h = {p: head_to_head(common, *p) for p in sre.head_to_head_pairs(cols)}
    steps = steps_tables(common, cols)
    tables, n_common = write_summary(os.path.join(out, "perceptual_summary.md"), reports,
                                     cols, names, lights, sets, args, h2h, steps, check)
    write_csv(os.path.join(out, "perceptual_long.csv"), reports, cols, names)

    held = lights["held_np"]
    report = {
        "columns": [{"key": c["key"], "display": names[c["key"]], "kind": c["kind"],
                     "tag": c.get("tag"), "dir": c.get("dir"), "step": c.get("step"),
                     "manifest_steps": c.get("manifest_steps")} for c in cols],
        "manifests": manifests,
        "lights": {"lpips_set": args.lpips_lights, "lpips_heldout_idx": sets["lpips"],
                   "lpips_lights": held[sets["lpips"]],
                   "flip_set": args.flip_lights, "flip_heldout_idx": sets["flip"],
                   "flip_lights": held[sets["flip"]],
                   "heldout_elevation_deg": [elev_deg(lights, j) for j in range(len(held))],
                   "heldout_min_sep_deg": lights["held_min_sep_deg"],
                   "sheet_lights": lights["sheet"]},
        "constants": {"lpips": "alex v0.1, inputs x * 2 - 1", "flip": "LDR, applyMagma=False",
                      "flip_parameters": metric.flip_params, "versions": metric.versions,
                      "identity_tol": IDENTITY_TOL, "response_min": RESPONSE_MIN,
                      "chunk": args.chunk, "device": str(device), "cudnn_tf32": False,
                      "torch_threads": torch.get_num_threads()},
        "metric_check": check,
        "n_common": n_common,
        "means": tables,
        "head_to_head": {"%s_vs_%s" % p: v for p, v in h2h.items()},
        "steps": steps,
        "materials": reports,
    }
    with open(os.path.join(out, "perceptual_report.json"), "w") as fh:
        json.dump(sre.clean(report), fh, indent=1)
    print("[perceptual] %d material(s), %d common -> %s" % (
        len(reports), n_common, os.path.join(out, "perceptual_summary.md")))


if __name__ == "__main__":
    main()
