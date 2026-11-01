#!/usr/bin/env python3
"""Render every method's PREDICTED SVBRDF with MatUp's own renderer and score it.

Runs in the matup container (torch 2.0, Python 3.10, cv2), after
deeptex_compare.py --maps all --scalar-collapse mean has written each arm:

    docker compose run --rm matup python /workspace/variants/deeptex/render_loss/eval/svbrdf_render_eval.py \\
        --arm matsynth_all=/workspace/runs/deeptex/eval/matsynth_all \\
        --arm render_r1_40k=/workspace/runs/deeptex/eval/render_r1_40k \\
        --out /workspace/runs/deeptex/render_eval/main

The container starts in /workspace/external/matup_egsr, so the script path and
--out must be absolute; an --out under external/ is refused (exit 2).

WHY THIS EXISTS
---------------
The "render" column of deeptex_arms_report.py is DeepTextures super-resolving
MatUp's LR RENDER (deeptex_compare.py:101, JOBS "render" <- render_lr.png). It
says nothing about whether the predicted MAPS render correctly - and that is the
question the render-loss arms (R0a/R0b/R1) ask. Here every column's five maps are
decoded exactly as MatUp decodes a MatSynth folder and rendered by MatUp's torch
renderer, so the GT row is MatUp's own render_gt.png and every other row is the
same pipeline fed a different map set.

COLUMNS (order: bilinear, matup, the --arm columns in the order given, gt)
  gt        GT/              matup_render.load_set(GT, "", True, True) + position3d
                             of the 16-bit GT height (matup_render.py:97-111)
  bilinear  SVBRDF/          matup_render.render_bilinear's map preparation,
                             returned instead of rendered (matup_render.py:114-140):
                             MatUp's own initialisation
  matup     run dir, "_opt"  render_matup's preparation (matup_render.py:143-160):
                             normal_opt is already in MatUp's frame (opengl=False),
                             height = the LR height bilinear x4
  dt_<tag>  <DIR>/<mat>/     load_set(<DIR>/<mat>, "", True, True) + position3d of
                             the arm's own 8-bit height
All are built at 512 and cropped by the border matup_postprocess.detect_border
measures on MatUp's opt maps without normal/height - the same call
deeptex_arms_report.py:174-177 makes - then get_renorm_clamped_maps
(data_utils.py:74-88), the order matup_render.render uses (:90-94).

RENDERING IS UNGUARDED (MatUp-faithful). Training guards n_z >= 1e-3 because
branchlessONB is 0/0 at n_z == 0 exactly (render_utils.py:96-97). A PNG-decoded
normal can never land there: an 8- or 16-bit level encodes 2b - (2^k - 1), an odd
number, never 0. A back-facing texel renders finitely (sign = -1) and dark, as in
MatUp, so render_gt.png etc. are reproduced as committed. Only an OFFSET-REMOVED
normal (below) can hit n_z == 0, on a measure-zero set; such texels get the
smallest positive float32 n_z instead of a NaN, and are counted (nz_zero_fixed,
expected 0 everywhere). The guard-relevant count n_z < 1e-3 is reported per
column (nz_below_min).

GT VALIDITY MASK. Every render metric runs over the texels whose GT normal passes
training's rule (render_tf.py:366-371): |2v - 1| >= 0.1 and unit n_z >= 1e-3,
measured on the stored GT normal PNG. 0844_syntetic_foam has back-facing GT
normals inside the crop; MatUp renders them black, training masks them, and so
does this. The mask is grey in the |diff| heatmaps.

LIGHTS
  train100  `1.4 * light_utils.generate_hemisphere_pts("Fibonacci", sample_count=100,
            near_z=False).float().to(device)` - main_optim.py:87-89 verbatim, i.e. the
            lights MatUp optimises against and render_tf trains against.
  heldout   matup_lights.heldout_lights(16): EVAL_LIGHT (norm 1, data_utils.py:111)
            plus 1.4 * fib(16) turned HELDOUT_AZIMUTH_DEG (43.25) about z. fib(n)
            shares its azimuths with fib(100), so the turn is what keeps every
            held-out light >= MIN_SEP_DEG (2.5; 2.89 at the closest) from every
            train100 direction; heldout_lights() raises otherwise. The measured
            separations go into report.json (lights.heldout_min_sep_deg).
  bands     matup_lights.elevation_bands (unit-z thresholds, never index ranges).
Lights are streamed --chunk at a time against a cached GT chunk; nothing of shape
(texels, 3, 100) is ever held per column.

VARIANTS
  raw   the maps as delivered.
  off   ORACLE OFFSET REMOVAL: per map and per channel, subtract mean(pred - gt)
        over the cropped image in the STORED encoding (sRGB baseColor, OpenGL-
        encoded normal, raw roughness / metallic / height in [0,1]), clip to
        [0,1], then decode with load_set's own steps (normal renormalised,
        roughness floored for the floor-at-load columns). This is
        deeptex_arms_report.offset_diagnostics (:122-141) carried into render
        space, with one departure: the normal is renormalised to unit length on
        both sides BEFORE its offset is taken, as read_img_as_float does to every
        normal.png (file_utils.py:87-88). The renderer sees only its direction,
        and the GT normals are stored short (mean |2v-1| 0.88 on acg_concrete_026,
        0.96 on th_brown_mud_leaves_001), so a stored-length offset would mostly
        be that length gap: removing it from a unit-length column (bilinear,
        MatUp) tilted perfect normals by up to 1.7 deg mean. The stored-length
        normal offset is still reported (normal_offset_255_stored).
        The DeepTextures recipe drifts in colour from checkpoint to
        checkpoint, and a render loss penalises a global offset directly, so a
        raw render win can be drift removal alone; P1 on "off" maps is the part
        that is not. Stored encodings: GT/DT read their PNGs; MatUp its _opt PNGs
        (normal G flipped to OpenGL, as matup_postprocess.py:170-173; roughness_opt
        is sqrt(alpha), data_utils.py:135-137) with the LR height bilinear x4;
        bilinear has no PNG, so its decoded maps are encoded back (gamma,
        y-flip, sqrt(alpha), z/0.01 + 0.5). With a zero offset "off" equals "raw"
        up to float rounding.

METRICS per (material, column, variant), all over GT-valid texels
  heldout_psnr / heldout_l21      P1 (variant off) and P1raw (variant raw): display
                                  PSNR with MSE over texels x lights x RGB, and
                                  MatUp's l2,1 (train.py:18-21, incl. its 1e-15 floor)
  train100_l21 / train100_psnr    P2: the training objective on unseen materials
  *_band_l21_{grazing,mid,high}   l2,1 per elevation band
  eval_psnr / eval_psnr_8bit      EVAL_LIGHT only, float and rint(x255) on both sides
  swap_l21_<map>                  held-out l2,1 of the GT with ONE map replaced by
                                  the prediction: each map's share of the error
  normal_angle_deg                mean angle between rendered (unit) normals
  render_grad / _pct_gt           matup_postprocess.sharpness |grad| of the EVAL_LIGHT
                                  render, quantised as write_image_as_png does
                                  (file_utils.py:112-125); 100 = as much detail as GT
  nz_below_min, nz_zero_fixed,    counts on the rendered normals (whole crop)
  zero_normals, metallic_mean
Materials are split non-metal (GT metallic all zero) / metal.

SELF-CHECK (exit 3). The EVAL_LIGHT renders of gt / bilinear / matup are compared
with the committed render_gt.png / render_bilinear.png / render_opt.png
(matup_postprocess.RENDER; written by matup_render.py:176-185 and by MatUp).
  gt, bilinear  <= 2/65535: the identical code wrote them, and
                write_image_as_png truncates (file_utils.py:115), i.e. <= 1 LSB.
  matup         <= 5e-4: render_opt.png was rendered by MatUp from its IN-MEMORY
                float maps; the 16-bit _opt PNGs read back here move the re-render
                by up to 3.3e-4 on these 12 materials (matup_render.py --verify,
                docs/RUNBOOK.md:131-132), so 2/65535 cannot hold for this column.
A failure means the map preparation here is wrong (normal convention, grid, crop),
and every number below would be meaningless.

OUTPUTS (all in --out): report.json, metrics_long.csv, summary.md,
per_material.md, render_<mat>.png (rows = methods, columns = EVAL_LIGHT, one
grazing, one mid, one high held-out light, and the EVAL_LIGHT |pred - GT| heatmap
at a FIXED gain), overview_render_pred.png (rows = materials, columns = methods,
EVAL_LIGHT). --no-sheets skips the PNGs. Those of an earlier run (these names
only) are removed from --out before anything is scored, and a missing or
unreadable GT / SVBRDF / _opt map is exit 2 - MatUp's own reader would exit 0.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

# Before anything from external/ is imported: Python 3.10 must not leave
# cpython-310.pyc files inside the submodule.
sys.dont_write_bytecode = True

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4)))
from variants.util import paths  # noqa: E402
REPO = paths.REPO
MATUP_ROOT = paths.use_matup()

from matup import data_utils, dataLoader, file_utils, light_utils, render_utils  # noqa: E402
from matup.renderer import Renderer  # noqa: E402

from variants.util import arms, matup_lights  # noqa: E402
from variants.matup.baseline import matup_outputs as mo  # noqa: E402
from variants.util.report import images as rimg, metrics as rmet, sheets as rsheet  # noqa: E402
from variants.matup.baseline.matup_outputs import newest_run  # noqa: E402,F401
from variants.matup.baseline import matup_render as mr  # noqa: E402

# ------------------------------------------------------------------- constants

# Training's GT validity rule (render_tf.py:125-127, applied at :366-371).
NZ_MIN = 1e-3
MASK_MIN_NORM = 0.1
NORMAL_NORM_FLOOR = 1e-4
L21_EPS = 1e-15                  # train.py:18-21
MIN_ROUGHNESS = 0.045            # dataLoader.py:39-40 (matup_render.py:68-72)
N_TRAIN = 100                    # main_optim.py:122 default --nb_lights
N_HELDOUT_FIB = 16
TINY = float(np.finfo(np.float32).tiny)

# Five maps in report order; SLOT is their index in load_set/position3d's list
# [albedo, metallic, normal, alpha, pos] (matup_render.py:76,109-110).
MAP_ORDER = ("baseColor", "normal", "roughness", "metallic", "height")
SLOT = {"baseColor": 0, "metallic": 1, "normal": 2, "roughness": 3, "height": 4}

VARIANTS = ("off", "raw")
VARIANT_TITLE = {"off": "offset-removed maps", "raw": "raw maps"}

U16 = 65535.0
SELF_TOL = {"gt": 2.0 / U16, "bilinear": 2.0 / U16, "matup": 5e-4}

HEAT_GAIN = 4.0                  # |pred - GT| x 4: 0.25 (64 display levels) saturates
CELL = 320                       # deeptex_arms_report.CELL / matup_montage --size
OV_CELL = 190                    # deeptex_arms_report.OV_CELL / matup_overview --size
SHEET_LEFT = 250
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORMAP = getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET)

# Arm names, column keys and comparison pairs come from the arm registry
# (variants/arms.json via variants/util/arms.py). A row says which method it is, not
# which folder it came from; an unknown tag shows as itself.
FIXED_NAMES = {"bilinear": "bilinear", "matup": "MatUp", "gt": "GT"}
FIXED_KINDS = ("bilinear", "matup")


# Scalars per (material, column, variant), in CSV / table order.
BAND_NAMES = ("grazing", "mid", "high")
SCALARS = (["heldout_psnr", "heldout_l21", "train100_l21", "train100_psnr"]
           + ["train100_band_l21_%s" % b for b in BAND_NAMES]
           + ["heldout_band_l21_%s" % b for b in BAND_NAMES]
           + ["eval_psnr", "eval_psnr_8bit", "normal_angle_deg", "render_grad",
              "render_grad_pct_gt", "nz_below_min", "nz_zero_fixed", "zero_normals",
              "metallic_mean"]
           + ["swap_l21_%s" % m for m in MAP_ORDER])


def die(code, msg):
    print("[fatal] %s" % msg)
    sys.exit(code)


# ---------------------------------------------------------------- names, tags

split_tag = arms.split_tag
display_name = arms.display_name
label_lines = arms.label_lines
step_key = arms.step_key


def is_arm(col):
    """An SR arm column (any --arm), as opposed to the built-in bilinear / MatUp."""
    return col["kind"] not in FIXED_KINDS


# ------------------------------------------------------------------- reading

def read_raw(path, multi):
    """A map PNG as STORED: float64 in [0,1], RGB, grey maps as one channel.

    file_utils.read_img_as_float (file_utils.py:64-89) minus its normal
    renormalisation at :87-88 - the offsets are taken on the stored values, as
    deeptex_arms_report.offset_diagnostics does (the normal's after unit_enc,
    see remove_offsets), and the GT validity mask on the stored length.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        die(2, "cannot read %s" % path)
    if not multi and img.ndim > 2:
        img = img[:, :, 0]                                    # file_utils.py:84-85
    return file_utils.process_image_as_float(img, multi)     # :54-62 (BGR->RGB, /255|/65535)


def require_readable(paths, where):
    """Exit 2 unless every map decodes. mr.load_set / read_img_as_float end in
    file_utils.exit_with_message on a missing or undecodable PNG, and that is
    exit(0) (file_utils.py:20-24, :80-82): a caller checking the exit code would
    take the run for a success."""
    bad = [p for p in paths
           if not os.path.isfile(p) or cv2.imread(p, cv2.IMREAD_UNCHANGED) is None]
    if bad:
        die(2, "%s: missing or unreadable: %s" % (where, ", ".join(bad)))


def enc_stored(path, suffix=""):
    rd = lambda name, multi: read_raw(os.path.join(path, name + suffix + ".png"), multi)  # noqa: E731
    return {"baseColor": rd("baseColor", True), "normal": rd("normal", True),
            "roughness": rd("roughness", False), "metallic": rd("metallic", False)}


def read_height(path):
    """float32 tensor, as matup_render.py:102-103 / :125-126 read it."""
    return torch.from_numpy(file_utils.read_img_as_float(path, False)).float()


def load_gt(mat_dir):
    """matup_render.render_source(..., "GT", ...) without the render (:97-111)."""
    path = os.path.join(mat_dir, "GT")
    albedo, metallic, normal, alpha = mr.load_set(path, "", True, True)
    h = read_height(os.path.join(path, "height.png"))
    pos = mr.position3d(h, albedo.shape[0])
    enc = enc_stored(path)
    enc["height"] = read_raw(os.path.join(path, "height.png"), False)
    return [albedo, metallic, normal, alpha, pos], enc, True


def load_dt(arm_dir, name):
    """A deeptex_compare.py output folder read exactly like a GT folder."""
    path = os.path.join(arm_dir, name)
    albedo, metallic, normal, alpha = mr.load_set(path, "", True, True)
    h = read_height(os.path.join(path, "height.png"))
    pos = mr.position3d(h, albedo.shape[0])
    enc = enc_stored(path)
    enc["height"] = read_raw(os.path.join(path, "height.png"), False)
    return [albedo, metallic, normal, alpha, pos], enc, True


def load_matup(mat_dir, run_dir):
    """matup_render.render_matup's map preparation (:143-160), not rendered."""
    albedo, metallic, normal, alpha = mr.load_set(run_dir, "_opt", False, False)  # :145
    size = albedo.shape[0]
    h_lr = file_utils.read_img_as_float(
        os.path.join(mat_dir, "SVBRDF", "height.png"), False)                    # :151-152
    z = mr.DISP_VAL * (h_lr - 0.5)                                                # :153
    z = cv2.resize(z, (size, size), interpolation=cv2.INTER_LINEAR)               # :154
    pos2d = data_utils.compute_2D_position_grid(size).float()                     # :155
    pos = torch.cat([pos2d, torch.from_numpy(z).float()[..., None]], dim=-1).float()  # :156

    enc = enc_stored(run_dir, "_opt")
    n = enc["normal"].copy()
    n[..., 1] = 1.0 - n[..., 1]          # DirectX -> OpenGL, matup_postprocess.py:170-173
    enc["normal"] = n
    enc["height"] = cv2.resize(h_lr, (size, size), interpolation=cv2.INTER_LINEAR)
    return [albedo, metallic, normal, alpha, pos], enc, False


def load_bilinear(mat_dir, hr):
    """matup_render.render_bilinear's map preparation (:114-140), not rendered.

    MatUp's initialisation: every map is decoded at 128 first and only then
    resized (dataLoader.py:105-128), the normal from its upsampled XY.
    """
    path = os.path.join(mat_dir, "SVBRDF")
    albedo, metallic, normal, alpha = mr.load_set(path, "", True, True)           # :124
    h = read_height(os.path.join(path, "height.png"))                             # :125-126
    z = mr.DISP_VAL * (h - 0.5)                                                   # :127

    def up(t):                                                                    # :129-130
        return torch.from_numpy(cv2.resize(
            t.numpy(), (hr, hr), interpolation=cv2.INTER_LINEAR)).float()

    albedo, metallic = up(albedo), up(metallic)[..., None]                        # :132
    alpha = up(alpha)[..., None]                                                  # :133
    normal = data_utils.normal_from_data_vector(up(normal[..., :2]))              # :134
    pos2d = data_utils.compute_2D_position_grid(hr).float()                       # :135
    pos = torch.cat([pos2d, up(z)[..., None]], dim=-1).float()                    # :136
    maps = [albedo, metallic, normal, alpha, pos]
    return maps, encode_maps(maps), False


def encode_maps(maps):
    """Decoded maps -> the stored encoding (only the bilinear column needs it:
    MatUp never writes its initialisation to disk). Inverses of load_set's steps."""
    albedo, metallic, normal, alpha, pos = maps
    n = normal.double().numpy().copy()
    n[..., 1] = -n[..., 1]                              # back to OpenGL (dataLoader.py:74-75)
    return {
        "baseColor": render_utils.gammaCorrection(albedo.double()).numpy(),  # inverse srgb2linrgb
        "normal": 0.5 * n + 0.5,
        "roughness": np.sqrt(alpha.double().numpy()[..., 0]),  # what roughness_opt stores
        "metallic": metallic.double().numpy()[..., 0],
        "height": pos[..., 2].double().numpy() / mr.DISP_VAL + 0.5,
    }


def unit_enc(e):
    """A normal in the stored encoding renormalised to unit length, as
    read_img_as_float does every normal.png (file_utils.py:87-88). Returns
    (encoding, zero-vector mask).

    normalize_np_norm divides with `where=` and no `out=` (data_utils.py:50), so a
    zero vector comes back as uninitialised memory; those texels are set to the
    zero vector (0.5) explicitly.
    """
    zero = ~np.any((2.0 * e - 1.0) != 0.0, axis=-1)
    nn = data_utils.normalize_np_norm(e)
    nn[zero] = 0.5
    return nn, zero


def decode_enc(enc, floor_roughness, pos2d):
    """Stored-encoding maps -> renderer maps by load_set's own steps
    (matup_render.py:53-83), for the offset-removed variant. Zero normals (see
    unit_enc) are counted.
    """
    albedo = file_utils.srgb2linrgb(torch.from_numpy(enc["baseColor"])).float()    # :62
    metallic = torch.from_numpy(enc["metallic"]).float()[..., None]               # :63
    nn, zero = unit_enc(enc["normal"])                                            # file_utils.py:87-88
    normal = dataLoader.unpack_normal(torch.from_numpy(nn), True).float()         # :64-65
    rough = torch.from_numpy(enc["roughness"]).float()                            # :67
    if floor_roughness:
        rough = torch.clamp(rough, min=MIN_ROUGHNESS)                             # :68-72
    alpha = torch.clamp(torch.pow(rough, 2.0),
                        min=Renderer.min_alpha, max=1.0).float()[..., None]       # :73-74
    h = torch.from_numpy(enc["height"]).float()
    z = (mr.DISP_VAL * (h - 0.5)).float()[..., None]                              # :82
    pos = torch.cat([pos2d, z], dim=-1).float()                                   # :83
    return [albedo, metallic, normal, alpha, pos], int(zero.sum())


def crop_enc(enc, b):
    return {k: mr.crop(v, b) for k, v in enc.items()}


def mean_diff(p, g):
    return (p - g).reshape(-1, p.shape[-1] if p.ndim == 3 else 1).mean(axis=0)


def remove_offsets(enc_pred, enc_gt, where):
    """Per map, per channel: subtract mean(pred - gt) over the crop, clip [0,1].

    The normal's offset is taken after unit_enc on BOTH sides: rendering keeps
    only its direction, and a mean over stored lengths would charge the gap
    between short GT normals and unit-length predictions as a z drift (module
    docstring, VARIANTS).

    Returns (offset-removed encodings, offsets in 8-bit levels, RGB order, the
    normal's offset on the stored lengths in 8-bit levels - diagnostic only).
    """
    out, off, stored = {}, {}, None
    for k in MAP_ORDER:
        p, g = enc_pred[k], enc_gt[k]
        if p.shape != g.shape:
            die(2, "%s: %s is %s but GT is %s" % (where, k, p.shape, g.shape))
        if k == "normal":
            stored = [float(v) * 255.0 for v in mean_diff(p, g)]
            p, g = unit_enc(p)[0], unit_enc(g)[0]
        o = mean_diff(p, g)
        out[k] = np.clip(p - (o if p.ndim == 3 else o[0]), 0.0, 1.0)
        off[k] = [float(v) * 255.0 for v in o]
    return out, off, stored


def gt_valid_mask(gt_dir, b):
    """Training's GT rule (render_tf.py:366-371) on the STORED GT normal."""
    n = 2.0 * read_raw(os.path.join(gt_dir, "normal.png"), True) - 1.0
    sq = (n * n).sum(axis=-1)
    nz_unit = n[..., 2] / np.sqrt(np.maximum(sq, NORMAL_NORM_FLOOR))
    return mr.crop((np.sqrt(sq) >= MASK_MIN_NORM) & (nz_unit >= NZ_MIN), b)


# newest_run: variants/compare/matup_runs.py (imported above; sre.newest_run and
# ev.newest_run keep resolving through this module's namespace).


# ------------------------------------------------------------------ rendering

def prepare(maps, b, device):
    """Crop, then MatUp's renormalise/clamp (data_utils.py:74-88) on the device -
    matup_render.render's order (:90-94). Fixes only n_z == 0 exactly (see the
    module docstring) and returns how many texels that touched."""
    m = data_utils.get_renorm_clamped_maps([mr.crop(t, b).to(device) for t in maps])
    nz = m[2][..., 2]
    zero = nz == 0
    k = int(zero.sum())
    if k:
        n = m[2].clone()
        n[..., 2] = torch.where(zero, torch.full_like(nz, TINY), nz)
        m[2] = n
    return m, k


def render_display(rmaps, lights, where):
    """(N, 3, L) display values: render_utils.batch_render_tn (:49-67) without its
    final reshape, keeping its finiteness assertion as an exit code."""
    flat = render_utils.reshape_maps_input_render(rmaps)
    raw = Renderer.torch_Render(*flat, lights, Renderer.z_vector)
    disp = render_utils.process_raw_render(raw)
    if not bool(torch.isfinite(disp).all()):
        die(4, "non-finite render: %s" % where)
    return disp


def spans(n, chunk):
    return [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]


def render_chunks(rmaps, lights, chunk, where, keep=()):
    out, kept = [], {}
    for lo, hi in spans(lights.shape[0], chunk):
        d = render_display(rmaps, lights[lo:hi], where)
        out.append(d)
        for j in keep:
            if lo <= j < hi:
                kept[j] = d[:, :, j - lo].cpu().numpy()
    return out, kept


def score(rmaps, lights, gt_chunks, mask, chunk, where, keep=()):
    """Per-light masked sums against cached GT chunks: (SSE over RGB, l2,1 terms).

    The texel-light error is train.py:18-21 verbatim (float32); only the masked
    sums over texels are taken in float64.
    """
    sse, l21s, kept = [], [], {}
    for (lo, hi), g in zip(spans(lights.shape[0], chunk), gt_chunks):
        d = render_display(rmaps, lights[lo:hi], where)
        e2 = torch.sum(torch.pow(d - g, 2.0), dim=1)                     # (N, l)
        sse.append((e2 * mask[:, None]).double().sum(dim=0))
        l21s.append((torch.sqrt(torch.clamp(e2, min=L21_EPS)) * mask[:, None])
                    .double().sum(dim=0))
        for j in keep:
            if lo <= j < hi:
                kept[j] = d[:, :, j - lo].cpu().numpy()
    return torch.cat(sse).cpu().numpy(), torch.cat(l21s).cpu().numpy(), kept


def eval_image(rmaps, light, where):
    h, w = rmaps[0].shape[:2]
    return render_display(rmaps, light, where).reshape(h, w, 3).cpu().numpy()


def psnr(mse, peak=1.0):
    """None for an exact match (texsr_metrics.full_reference's JSON convention)."""
    return None if mse <= 0 else float(10.0 * math.log10(peak * peak / mse))


def light_block(prefix, sse, l21s, n_valid, bands):
    n_l = len(sse)
    out = {prefix + "_psnr": psnr(float(sse.sum()) / (n_valid * n_l * 3.0)),
           prefix + "_l21": float(l21s.sum()) / (n_valid * n_l),
           prefix + "_l21_per_light": [float(v) / n_valid for v in l21s]}
    for b in BAND_NAMES:
        idx = bands[b]
        out["%s_band_l21_%s" % (prefix, b)] = (
            float(l21s[idx].sum()) / (n_valid * len(idx)) if len(idx) else None)
    return out


def to_u16(img):
    """write_image_as_png's quantisation, truncation included (file_utils.py:114-115)."""
    return np.array(U16 * np.clip(img, 0.0, 1.0), dtype=np.uint16)


def column_metrics(rmaps, ctx, args, where, keep=()):
    nv = ctx["n_valid"]
    h_sse, h_l21, kept = score(rmaps, ctx["held"], ctx["gt_held"], ctx["mask"],
                               args.chunk, where + " heldout", keep)
    t_sse, t_l21, _ = score(rmaps, ctx["train"], ctx["gt_train"], ctx["mask"],
                            args.chunk, where + " train100")
    m = {}
    m.update(light_block("heldout", h_sse, h_l21, nv, ctx["bands_held"]))
    m.update(light_block("train100", t_sse, t_l21, nv, ctx["bands_train"]))

    n_held = ctx["held"].shape[0]
    for k in MAP_ORDER:
        swapped = list(ctx["gt_rmaps"])
        swapped[SLOT[k]] = rmaps[SLOT[k]]
        _, s_l21, _ = score(swapped, ctx["held"], ctx["gt_held"], ctx["mask"],
                            args.chunk, "%s swap %s" % (where, k))
        m["swap_l21_" + k] = float(s_l21.sum()) / (nv * n_held)

    img = eval_image(rmaps, ctx["eval"], where + " eval")
    mk = ctx["mask_hw"]
    d = (img.astype(np.float64) - ctx["gt_eval"].astype(np.float64))[mk]
    m["eval_psnr"] = psnr(float((d * d).mean()))
    q = lambda x: np.rint(np.clip(x, 0.0, 1.0) * 255.0)  # noqa: E731
    d8 = (q(img) - q(ctx["gt_eval"]))[mk]
    m["eval_psnr_8bit"] = psnr(float((d8 * d8).mean()), 255.0)
    m["render_grad"] = rmet.sharpness(to_u16(img))[0]
    m["render_grad_pct_gt"] = (100.0 * m["render_grad"] / ctx["gt_grad"]
                               if ctx["gt_grad"] else None)

    a = rmaps[2].reshape(-1, 3).double()
    g = ctx["gt_rmaps"][2].reshape(-1, 3).double()
    ang = torch.rad2deg(torch.acos(torch.clamp((a * g).sum(dim=1), -1.0, 1.0)))
    m["normal_angle_deg"] = float(ang.cpu().numpy()[ctx["mask_flat"]].mean())
    m["nz_below_min"] = int((rmaps[2][..., 2] < NZ_MIN).sum())
    m["metallic_mean"] = float(rmaps[1].double().mean())
    return m, img, kept


# ---------------------------------------------------------------- self-check

def self_check(col, ours, run_dir, border, full):
    """Our EVAL_LIGHT render vs the committed PNG of the same column.

    Every image here is a centre crop of the same full x full grid, so the two
    are aligned by trimming the wider one; render_opt.png keeps MatUp's zero rim
    and is first cropped by the border, as deeptex_arms_report.py:216-217 does.
    """
    path = os.path.join(run_dir, mo.RENDER[col])
    if not os.path.exists(path):
        die(3, "self-check: %s is missing - run matup_render.py first" % path)
    theirs = rimg.read(path)[..., ::-1].astype(np.float64) / U16    # BGR -> RGB
    if theirs.shape[0] == full:
        theirs = mr.crop(theirs, border)
        b_t = border
    else:
        b_t = (full - theirs.shape[0]) // 2
    b = max(border, b_t)
    a = mr.crop(ours.astype(np.float64), b - border)
    t = mr.crop(theirs, b - b_t)
    if a.shape != t.shape:
        die(3, "self-check: %s is %s, ours %s after alignment" % (path, t.shape, a.shape))
    diff = np.abs(a - t)
    rec = {"file": path, "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
           "tol": SELF_TOL[col], "pass": bool(diff.max() <= SELF_TOL[col])}
    if not rec["pass"]:
        die(3, "self-check FAILED for %s: max |ours - committed| = %.3g > %.3g (%s). "
               "The map preparation here does not reproduce MatUp's render."
            % (col, rec["max_abs"], rec["tol"], path))
    return rec


# ------------------------------------------------------------------- pictures

def to_cell(img, size):
    """Float RGB render -> the montage cell (8-bit BGR), via the committed sheets'
    to_display; quantised like the PNG MatUp would have written."""
    return rimg.to_display(np.ascontiguousarray(to_u16(img)[..., ::-1]), size)


def heat_cell(img, gt, mask_hw, size):
    d = np.abs(img.astype(np.float64) - gt.astype(np.float64)).mean(axis=2) * HEAT_GAIN
    u8 = (np.clip(d, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, COLORMAP)
    bgr[~mask_hw] = (128, 128, 128)
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)


def draw_lines(img, lines, x, y, width, start, colour):
    """deeptex_arms_report.draw_lines (:236-240)."""
    for i, t in enumerate(lines):
        s = rsheet.fit(t, width, start if i == 0 else start * 0.85)
        cv2.putText(img, t, (x, y + i * int(26 * start + 6)), FONT, s, colour, 1, cv2.LINE_AA)


def write_render_sheet(path, title, notes, heads, rows):
    """deeptex_arms_report.write_montage's layout (:243-265): methods down,
    renders across."""
    pad, head = rsheet.GRID.pad, rsheet.GRID.head
    top = rsheet.GRID.title + 18 * len(notes)
    w = SHEET_LEFT + len(heads) * (CELL + pad) + pad
    h = top + head + len(rows) * (CELL + pad) + pad
    sheet = np.full((h, w, 3), 18, np.uint8)
    cv2.putText(sheet, title, (10, 22), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for i, t in enumerate(notes):
        cv2.putText(sheet, t, (10, 22 + 18 * (i + 1)), FONT, rsheet.fit(t, w - 20, 0.45),
                    (200, 200, 200), 1, cv2.LINE_AA)
    for c, t in enumerate(heads):
        x = SHEET_LEFT + c * (CELL + pad)
        cv2.putText(sheet, t, (x + 4, top + 22), FONT, rsheet.fit(t, CELL - 8, 0.5),
                    (210, 210, 210), 1, cv2.LINE_AA)
    for r, (name, cells) in enumerate(rows):
        y = top + head + r * (CELL + pad)
        draw_lines(sheet, label_lines(name), 10, y + CELL // 2 - 8, SHEET_LEFT - 18, 0.62,
                   (210, 210, 210))
        for c, cell in enumerate(cells):
            x = SHEET_LEFT + c * (CELL + pad)
            sheet[y:y + CELL, x:x + CELL] = cell if cell is not None else rsheet.placeholder(CELL, [])
    rimg.write(path, sheet)


def write_overview(path, rows, keys, names):
    """deeptex_arms_report.write_overview's layout (:268-293): materials down,
    methods across."""
    pad, left, top, head = rsheet.TABLE.pad, rsheet.TABLE.left, rsheet.TABLE.title, 46
    w = left + len(keys) * (OV_CELL + pad) + pad
    h = top + head + len(rows) * (OV_CELL + pad) + pad
    sheet = np.full((h, w, 3), 18, np.uint8)
    cv2.putText(sheet, "Renders of the predicted SVBRDF (MatUp renderer, EVAL_LIGHT)  -  %d materials"
                % len(rows), (10, 24), FONT, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    for c, key in enumerate(keys):
        x = left + c * (OV_CELL + pad)
        draw_lines(sheet, label_lines(names[key]), x + 3, top + 18, OV_CELL - 6, 0.5,
                   (215, 215, 215))
    for r, (name, cells) in enumerate(rows):
        y = top + head + r * (OV_CELL + pad)
        cv2.putText(sheet, name, (8, y + OV_CELL // 2), FONT,
                    rsheet.fit(name, left - 16, 0.46), (215, 215, 215), 1, cv2.LINE_AA)
        for c, key in enumerate(keys):
            x = left + c * (OV_CELL + pad)
            cell = cells.get(key)
            if cell is None:
                cell = np.full((OV_CELL, OV_CELL, 3), 30, np.uint8)
                cv2.putText(cell, "n/a", (OV_CELL // 2 - 18, OV_CELL // 2), FONT, 0.5,
                            (110, 110, 110), 1, cv2.LINE_AA)
            sheet[y:y + OV_CELL, x:x + OV_CELL] = cell
    rimg.write(path, sheet)


# --------------------------------------------------------------- per material

def process(mat_dir, cols, args, device, lights):
    name = os.path.basename(mat_dir.rstrip("/"))
    run_dir = newest_run(mat_dir, name, args.run)
    if not run_dir:
        print("  %-28s SKIP (no MatUp output)" % name)
        return None, None, None
    # gt, bilinear and matup are always columns; a hole in their inputs is fatal.
    require_readable([os.path.join(mat_dir, sub, m + ".png")
                      for sub in ("GT", "SVBRDF") for m in MAP_ORDER]
                     + [os.path.join(run_dir, m + "_opt.png")
                        for m in MAP_ORDER if m != "height"], name)

    # Border: deeptex_arms_report.py:166-177 - MatUp's zero rim, never measured
    # on normal (mid-grey rim) or the pre-cropped height.
    opts = {}
    for m, (_, opt_name) in mo.MAPS.items():
        p = os.path.join(run_dir, opt_name)
        if os.path.exists(p):
            opts[m] = rimg.read(p)
    border, sides = mo.detect_border(
        [a for m, a in opts.items() if m != "normal" and m not in mo.PRECROPPED])

    gt_maps, gt_enc_full, _ = load_gt(mat_dir)
    full = gt_maps[0].shape[0]
    side = full - 2 * border
    pos2d = mr.crop(data_utils.compute_2D_position_grid(full).float(), border)
    mask_hw = gt_valid_mask(os.path.join(mat_dir, "GT"), border)
    n_valid = int(mask_hw.sum())
    if n_valid == 0:
        die(2, "%s: no GT texel passes the validity rule" % name)
    metal = bool(read_raw(os.path.join(mat_dir, "GT", "metallic.png"), False).max() > 0)

    rep = {"material": name, "run": os.path.basename(run_dir), "border": border,
           "border_sides": list(sides), "size": side, "metal": metal,
           "n_valid": n_valid, "valid_frac": n_valid / float(side * side),
           "self_check": {}, "columns": {}, "missing": []}

    gt_rmaps, gt_fix = prepare(gt_maps, border, device)
    gt_enc = crop_enc(gt_enc_full, border)
    gt_eval = eval_image(gt_rmaps, lights["eval"], "%s gt eval" % name)
    rep["self_check"]["gt"] = self_check("gt", gt_eval, run_dir, border, full)
    sheet_idx = sorted(lights["sheet"].values())
    gt_held, gt_kept = render_chunks(gt_rmaps, lights["held"], args.chunk,
                                     "%s gt heldout" % name, sheet_idx)
    gt_train, _ = render_chunks(gt_rmaps, lights["train"], args.chunk, "%s gt train100" % name)
    gt_grad = rmet.sharpness(to_u16(gt_eval))[0]
    rep["columns"]["gt"] = {"raw": {
        "render_grad": gt_grad, "nz_below_min": int((gt_rmaps[2][..., 2] < NZ_MIN).sum()),
        "nz_zero_fixed": gt_fix, "metallic_mean": float(gt_rmaps[1].double().mean())}}

    ctx = {"n_valid": n_valid, "mask_hw": mask_hw, "mask_flat": mask_hw.reshape(-1),
           "mask": torch.from_numpy(mask_hw.reshape(-1).astype(np.float32)).to(device),
           "held": lights["held"], "train": lights["train"], "eval": lights["eval"],
           "bands_held": lights["bands_held"], "bands_train": lights["bands_train"],
           "gt_rmaps": gt_rmaps, "gt_held": gt_held, "gt_train": gt_train,
           "gt_eval": gt_eval, "gt_grad": gt_grad}

    def sheet_row(img, kept):
        cells = [to_cell(img, CELL)]
        for b in BAND_NAMES:
            j = lights["sheet"].get(b)
            cells.append(None if j is None else to_cell(kept[j].reshape(side, side, 3), CELL))
        cells.append(heat_cell(img, gt_eval, mask_hw, CELL))
        return cells

    sheet_rows, ov_cells = {}, {}
    line = []
    for col in cols:
        key = col["key"]
        if col["kind"] == "bilinear":
            maps, enc, floor = load_bilinear(mat_dir, full)
        elif col["kind"] == "matup":
            maps, enc, floor = load_matup(mat_dir, run_dir)
        else:
            d = os.path.join(col["dir"], name)
            need = [os.path.join(d, m + ".png") for m in MAP_ORDER]
            lost = [p for p in need if not os.path.exists(p)]
            if lost:
                print("  ! %s: arm %s has no %s" % (name, col["tag"], ", ".join(
                    os.path.basename(p) for p in lost)))
                rep["missing"].append(key)
                continue
            require_readable(need, "%s / %s" % (name, key))
            maps, enc, floor = load_dt(col["dir"], name)
        if maps[0].shape[0] != full or any(t.shape[:2] != maps[0].shape[:2] for t in maps):
            die(2, "%s / %s: map sizes %s, GT is %d" % (
                name, key, [tuple(t.shape) for t in maps], full))

        where = "%s %s" % (name, key)
        entry = {}
        raw, fix = prepare(maps, border, device)
        m, img, kept = column_metrics(raw, ctx, args, where + " raw", sheet_idx)
        m["nz_zero_fixed"], m["zero_normals"] = fix, 0
        entry["raw"] = m
        if col["kind"] in SELF_TOL:
            rep["self_check"][col["kind"]] = self_check(col["kind"], img, run_dir, border, full)
        sheet_rows[key] = sheet_row(img, kept)
        ov_cells[key] = to_cell(img, OV_CELL)
        del raw

        enc_off, offsets, n_stored = remove_offsets(crop_enc(enc, border), gt_enc, where)
        off_maps, zero_n = decode_enc(enc_off, floor, pos2d)
        off, fix = prepare(off_maps, 0, device)
        m, _, _ = column_metrics(off, ctx, args, where + " off")
        m["nz_zero_fixed"], m["zero_normals"] = fix, zero_n
        m["offset_255"] = offsets
        m["normal_offset_255_stored"] = n_stored
        entry["off"] = m
        del off
        rep["columns"][key] = entry
        line.append("%s %s/%s" % (col["short"], fmt(entry["off"]["heldout_psnr"], "%.2f"),
                                  fmt(entry["raw"]["heldout_psnr"], "%.2f")))

    sheet_rows["gt"] = [to_cell(gt_eval, CELL)] + [
        None if lights["sheet"].get(b) is None else
        to_cell(gt_kept[lights["sheet"][b]].reshape(side, side, 3), CELL)
        for b in BAND_NAMES] + [heat_cell(gt_eval, gt_eval, mask_hw, CELL)]
    ov_cells["gt"] = to_cell(gt_eval, OV_CELL)

    sc = rep["self_check"]
    print("  %-28s %s border=%d valid=%.4f%s self-check(LSB) %s" % (
        name, rep["run"], border, rep["valid_frac"], " metal" if metal else "",
        " ".join("%s=%.1f" % (c, sc[c]["max_abs"] * U16) for c in ("gt", "bilinear", "matup")
                 if c in sc)))
    print("      P1 off/raw PSNR: " + "  ".join(line))
    return rep, sheet_rows, ov_cells


# ----------------------------------------------------------------- aggregation

def fmt(v, f):
    return "-" if v is None else f % v


def common_materials(reports, keys):
    """Materials on which every non-GT column was scored (deeptex_arms_report
    common_materials, :298-311): every mean in a table is over the same set."""
    return [r for r in reports
            if all(k in r["columns"] and r["columns"][k]["off"]["heldout_psnr"] is not None
                   and r["columns"][k]["raw"]["heldout_psnr"] is not None for k in keys)]


def mean_of(reports, key, variant, metric):
    xs = [r["columns"][key][variant].get(metric) for r in reports]
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def head_to_head_pairs(cols):
    """Registry pairs (same step), then every arm vs its registry reference (same
    step, else the reference's unsuffixed tag), then every arm vs bilinear / MatUp."""
    dts = [c for c in cols if is_arm(c)]
    by = {(c["base"], c["step"]): c["key"] for c in dts}
    key_of = {c["tag"]: c["key"] for c in dts}
    keys = {c["key"] for c in cols}
    pairs = [(key_of[a], key_of[b]) for a, b in arms.pairs([c["tag"] for c in dts])]
    for c in dts:
        ref_base = arms.reference(c["tag"])
        if not ref_base or c["base"] == ref_base:
            continue
        ref = by.get((ref_base, c["step"])) or by.get((ref_base, ""))
        if ref:
            pairs.append((c["key"], ref))
    for c in dts:
        for other in ("bilinear", "matup"):
            if other in keys:
                pairs.append((c["key"], other))
    seen, out = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def head_to_head(reports, a, b):
    """Materials where a beats b (strictly) on P1, P1 l2,1 and P1raw, plus the mean
    paired difference a - b."""
    out = {"n": 0, "p1": 0, "p1_l21": 0, "p1raw": 0, "n_nonmetal": 0, "p1_nonmetal": 0,
           "mean_diff_p1_db": None, "mean_diff_p1raw_db": None, "mean_diff_p1_l21": None}
    d1, dr, dl = [], [], []
    for r in reports:
        if a not in r["columns"] or b not in r["columns"]:
            continue
        ea, eb = r["columns"][a], r["columns"][b]
        pa, pb = ea["off"]["heldout_psnr"], eb["off"]["heldout_psnr"]
        ra, rb = ea["raw"]["heldout_psnr"], eb["raw"]["heldout_psnr"]
        if None in (pa, pb, ra, rb):
            continue
        out["n"] += 1
        out["p1"] += pa > pb
        out["p1raw"] += ra > rb
        out["p1_l21"] += ea["off"]["heldout_l21"] < eb["off"]["heldout_l21"]
        if not r["metal"]:
            out["n_nonmetal"] += 1
            out["p1_nonmetal"] += pa > pb
        d1.append(pa - pb)
        dr.append(ra - rb)
        dl.append(ea["off"]["heldout_l21"] - eb["off"]["heldout_l21"])
    if d1:
        out["mean_diff_p1_db"] = float(np.mean(d1))
        out["mean_diff_p1raw_db"] = float(np.mean(dr))
        out["mean_diff_p1_l21"] = float(np.mean(dl))
    for k in ("p1", "p1_l21", "p1raw", "p1_nonmetal"):
        out[k] = int(out[k])
    return out


def steps_tables(common, cols):
    """Per base arm with step-suffixed tags: P1 / P1raw mean at each step, the mean
    over steps and the within-run range - the inputs of the plan's tau."""
    dts = [c for c in cols if is_arm(c) and c["step"]]
    bases = []
    for c in dts:
        if c["base"] not in bases:
            bases.append(c["base"])
    steps = sorted({c["step"] for c in dts}, key=step_key)
    key_of = {(c["base"], c["step"]): c["key"] for c in dts}
    out = {}
    for label, variant, subset in (("P1 PSNR (off)", "off", "all"),
                                   ("P1raw PSNR (raw)", "raw", "all"),
                                   ("P1 PSNR (off), non-metal", "off", "nonmetal")):
        rs = common if subset == "all" else [r for r in common if not r["metal"]]
        rows = {}
        for base in bases:
            vals = {s: mean_of(rs, key_of[(base, s)], variant, "heldout_psnr")
                    for s in steps if (base, s) in key_of}
            got = [v for v in vals.values() if v is not None]
            rows[base] = {"steps": vals,
                          "mean": float(np.mean(got)) if got else None,
                          "range": float(max(got) - min(got)) if len(got) > 1 else None}
        out[label] = {"steps": steps, "rows": rows, "n": len(rs)}
    return out


def write_summary(path, reports, cols, names, lights, h2h, steps):
    keys = [c["key"] for c in cols]
    common = common_materials(reports, keys)
    nonmetal = [r for r in common if not r["metal"]]
    metal = [r for r in common if r["metal"]]
    L = ["# Renders of the predicted SVBRDF: %d MatSynth evaluation materials" % len(reports),
         "",
         "Every column's five maps are decoded the way MatUp decodes a MatSynth folder and",
         "rendered by MatUp's own torch renderer (UNGUARDED, as MatUp renders). Errors are",
         "against the render of the GT maps, over texels whose GT normal passes training's",
         "validity rule (|2v-1| >= 0.1, unit n_z >= 1e-3).",
         "",
         "- **held-out** = EVAL_LIGHT + 1.4 x fib(16) turned 43.25 deg about z (%d lights, none within %.2f deg of a training direction);" % (len(lights["held_np"]), float(np.min(lights["held_min_sep_deg"]))),
         "  **train-100** = MatUp's 1.4 x fib(100), the lights the render loss uses.",
         "- **P1** = held-out PSNR / l2,1 on OFFSET-REMOVED maps (primary); **P1raw** = the",
         "  same on the raw maps; **P2** = train-100 l2,1. Offset removal subtracts each map's",
         "  per-channel mean error vs GT in the stored encoding (the normal renormalised to",
         "  unit length on both sides first: only its direction renders), so P1 is what is",
         "  left after a constant drift is forgiven.",
         "- PSNR: MSE over texels x lights x RGB in display space ([0,1]). l2,1: MatUp's loss",
         "  (mean over texel-lights of the RGB L2 norm). Lower l2,1 is better.",
         "- `n`: materials every column was scored on; every mean below is over that set",
         "  (%d non-metal = GT metallic all zero, %d metal)." % (len(nonmetal), len(metal)),
         "- bands: train-100 lights by elevation, grazing < 20 deg (%d), high > 60 deg (%d),"
         % (len(lights["bands_train"]["grazing"]), len(lights["bands_train"]["high"])),
         "  mid otherwise (%d)." % len(lights["bands_train"]["mid"]),
         "- `|grad| %GT`: matup_postprocess.sharpness of the EVAL_LIGHT render as a % of the",
         "  GT render's; PSNR rewards blur, this does not.",
         "- self-check: EVAL_LIGHT renders of GT / bilinear / MatUp reproduce the committed",
         "  render_gt / render_bilinear / render_opt PNGs (see the table at the end).",
         "",
         "Columns: " + ", ".join("`%s` = %s" % (c["key"], names[c["key"]]) + (
             " (%s)" % c["dir"] if c.get("dir") else "") for c in cols) + ".", ""]

    tables = {}
    for v in VARIANTS:
        L += ["## %s - %s" % ("P1" if v == "off" else "P1raw", VARIANT_TITLE[v]), "",
              "| column | n | held-out PSNR | held-out l2,1 | train-100 l2,1 | train-100 PSNR "
              "| grazing l2,1 | mid l2,1 | high l2,1 | eval PSNR | eval PSNR 8-bit "
              "| normal ° | \\|grad\\| %GT | n_z<1e-3 | metallic |",
              "|---|" + "---|" * 14]
        tables[v] = {}
        for c in cols:
            k = c["key"]
            mv = {s: mean_of(common, k, v, s) for s in SCALARS}
            tables[v][k] = mv
            L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                names[k], len(common), fmt(mv["heldout_psnr"], "%.2f"),
                fmt(mv["heldout_l21"], "%.4f"), fmt(mv["train100_l21"], "%.4f"),
                fmt(mv["train100_psnr"], "%.2f"), fmt(mv["train100_band_l21_grazing"], "%.4f"),
                fmt(mv["train100_band_l21_mid"], "%.4f"), fmt(mv["train100_band_l21_high"], "%.4f"),
                fmt(mv["eval_psnr"], "%.2f"), fmt(mv["eval_psnr_8bit"], "%.2f"),
                fmt(mv["normal_angle_deg"], "%.2f"), fmt(mv["render_grad_pct_gt"], "%.0f%%"),
                fmt(mv["nz_below_min"], "%.0f"), fmt(mv["metallic_mean"], "%.3f")))
        L += ["", "Non-metal / metal split:", "",
              "| column | non-metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° "
              "| metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° |",
              "|---|" + "---|" * 10]
        for c in cols:
            k = c["key"]
            cells = []
            for rs in (nonmetal, metal):
                cells += ["%d" % len(rs)] + [fmt(mean_of(rs, k, v, s), f) for s, f in (
                    ("heldout_psnr", "%.2f"), ("heldout_l21", "%.4f"),
                    ("train100_l21", "%.4f"), ("normal_angle_deg", "%.2f"))]
            L.append("| %s | %s |" % (names[k], " | ".join(cells)))
        L += ["", "Map-swap attribution - held-out l2,1 of the GT with one map replaced by the",
              "column's (the full-prediction value last):", "",
              "| column | " + " | ".join(MAP_ORDER) + " | all five |",
              "|---|" + "---|" * (len(MAP_ORDER) + 1)]
        for c in cols:
            k = c["key"]
            L.append("| %s | %s | %s |" % (names[k], " | ".join(
                fmt(tables[v][k]["swap_l21_" + m], "%.4f") for m in MAP_ORDER),
                fmt(tables[v][k]["heldout_l21"], "%.4f")))
        L.append("")

    L += ["## Head-to-head", "",
          "Materials where the first column is strictly better; mean paired difference",
          "(first - second) over the same materials.", ""]
    for (a, b), hh in h2h.items():
        if not hh["n"]:
            continue
        L.append("- %s vs %s: P1 %d/%d (non-metal %d/%d), P1 l2,1 %d/%d, P1raw %d/%d; "
                 "mean diff P1 %s dB, P1raw %s dB, P1 l2,1 %s" % (
                     names[a], names[b], hh["p1"], hh["n"], hh["p1_nonmetal"],
                     hh["n_nonmetal"], hh["p1_l21"], hh["n"], hh["p1raw"], hh["n"],
                     fmt(hh["mean_diff_p1_db"], "%+.3f"), fmt(hh["mean_diff_p1raw_db"], "%+.3f"),
                     fmt(hh["mean_diff_p1_l21"], "%+.5f")))
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
                    arms.base_display(base),
                    " | ".join(fmt(row["steps"].get(s), "%.3f") for s in t["steps"]),
                    fmt(row["mean"], "%.3f"), fmt(row["range"], "%.3f")))
            L.append("")

    L += ["## Self-check (max |ours - committed| in 16-bit LSB)", "",
          "Tolerance: GT and bilinear %.1f LSB (same code wrote them; PNG truncation), "
          "MatUp %.1f LSB (render_opt.png came from MatUp's in-memory maps)."
          % (SELF_TOL["gt"] * U16, SELF_TOL["matup"] * U16), "",
          "| material | GT | bilinear | MatUp |", "|---|---|---|---|"]
    for r in reports:
        sc = r["self_check"]
        L.append("| %s | %s |" % (r["material"], " | ".join(
            fmt(sc[c]["max_abs"] * U16 if c in sc else None, "%.2f")
            for c in ("gt", "bilinear", "matup"))))
    fixes = [(r["material"], k, v, r["columns"][k][v]["nz_zero_fixed"],
              r["columns"][k][v].get("zero_normals", 0))
             for r in reports for k in keys if k in r["columns"] for v in VARIANTS
             if r["columns"][k][v]["nz_zero_fixed"] or r["columns"][k][v].get("zero_normals")]
    L += ["", "Texels whose rendered normal had n_z == 0 exactly (MatUp's ONB is 0/0 there; "
          "given the smallest positive n_z): %s." % (
              "none" if not fixes else "; ".join("%s %s %s: %d (zero vectors %d)" % f
                                                 for f in fixes)), ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return tables, len(common)


def write_per_material(path, reports, cols, names):
    L = ["# Per-material numbers", "",
         "Held-out lights, GT-valid texels. Cells are `PSNR / l2,1`.", ""]
    for v in VARIANTS:
        L += ["## %s - %s" % ("P1" if v == "off" else "P1raw", VARIANT_TITLE[v]), "",
              "| material | metal | " + " | ".join(names[c["key"]] for c in cols) + " |",
              "|---|---|" + "---|" * len(cols)]
        for r in reports:
            cells = []
            for c in cols:
                e = r["columns"].get(c["key"])
                cells.append("-" if e is None else "%s / %s" % (
                    fmt(e[v]["heldout_psnr"], "%.2f"), fmt(e[v]["heldout_l21"], "%.4f")))
            L.append("| %s | %s | %s |" % (r["material"], "yes" if r["metal"] else "",
                                           " | ".join(cells)))
        L.append("")
    L += ["## Normal angle (degrees, raw / offset-removed) and render |grad| %GT (raw)", "",
          "| material | " + " | ".join(names[c["key"]] for c in cols) + " |",
          "|---|" + "---|" * len(cols)]
    for r in reports:
        cells = []
        for c in cols:
            e = r["columns"].get(c["key"])
            cells.append("-" if e is None else "%s / %s, %s" % (
                fmt(e["raw"]["normal_angle_deg"], "%.1f"), fmt(e["off"]["normal_angle_deg"], "%.1f"),
                fmt(e["raw"]["render_grad_pct_gt"], "%.0f%%")))
        L.append("| %s | %s |" % (r["material"], " | ".join(cells)))
    L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def write_csv(path, reports, cols, names):
    fields = (["material", "metal", "column", "display", "tag", "step", "variant",
               "n_valid", "valid_frac"] + SCALARS
              + ["offset_255_%s" % m for m in MAP_ORDER] + ["normal_offset_255_stored"])
    meta = {c["key"]: c for c in cols}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in reports:
            for k, e in r["columns"].items():
                if k == "gt":
                    continue
                c = meta[k]
                for v in VARIANTS:
                    row = {"material": r["material"], "metal": r["metal"], "column": k,
                           "display": names[k], "tag": c.get("tag") or "",
                           "step": c.get("step") or "", "variant": v,
                           "n_valid": r["n_valid"], "valid_frac": r["valid_frac"]}
                    row.update({s: e[v].get(s) for s in SCALARS})
                    for m in MAP_ORDER:
                        o = e[v].get("offset_255", {}).get(m)
                        row["offset_255_%s" % m] = None if o is None else ";".join(
                            "%.2f" % x for x in o)
                    o = e[v].get("normal_offset_255_stored")
                    row["normal_offset_255_stored"] = None if o is None else ";".join(
                        "%.2f" % x for x in o)
                    w.writerow(row)


def clean(o):
    """numpy -> Python, non-finite -> None: json.dump would write the bare tokens
    NaN/Infinity, which are not JSON (texsr_metrics.full_reference)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ------------------------------------------------------------------------ main

def check_out(path):
    if not os.path.isabs(path):
        die(2, "--out must be absolute (the matup container starts in %s), got %r"
            % (MATUP_ROOT, path))
    out = os.path.normpath(path)
    for ext in {os.path.normpath(os.path.join(REPO, "external")),
                os.path.normpath(MATUP_ROOT), os.path.normpath("/workspace/external")}:
        if out == ext or out.startswith(ext + os.sep):
            die(2, "--out %s is under %s; the submodules must stay clean" % (out, ext))
    return out


def clear_stale(out, mats):
    """Remove this script's own outputs of an earlier run from --out before
    anything is scored, so a run that dies, or skips a material, never leaves an
    older summary or sheet looking current (run_deeptex.ps1 collects them by
    name). Only names this run would write are touched - never a wildcard, since
    --out could hold MatUp's render_gt/_opt/_bilinear/_lr.png."""
    names = ["report.json", "summary.md", "per_material.md", "metrics_long.csv",
             "overview_render_pred.png"]
    names += ["render_%s.png" % os.path.basename(m.rstrip("/")) for m in mats]
    gone = 0
    for n in names:
        p = os.path.join(out, n)
        if n not in mo.RENDER.values() and os.path.isfile(p):
            os.remove(p)
            gone += 1
    if gone:
        print("[eval] removed %d output file(s) of an earlier run from %s" % (gone, out))


def parse_arms(specs):
    cols, seen = [], set()
    for s in specs:
        tag, _, d = s.partition("=")
        if not tag or not d or not os.path.isdir(d):
            die(2, "--arm wants TAG=existing_dir, got %r" % s)
        if tag in seen or tag in FIXED_NAMES:
            die(2, "--arm tag %r is used twice or clashes with a built-in column" % tag)
        seen.add(tag)
        base, step = split_tag(tag)
        key = arms.column_key(tag)
        cols.append({"key": key, "kind": key[:key.index("_")], "tag": tag, "dir": os.path.normpath(d),
                     "base": base, "step": step, "short": tag})
    return cols


def build_lights(device):
    held_np = matup_lights.heldout_lights(N_HELDOUT_FIB)
    # main_optim.py:87-89 verbatim (cast first, multiply in float32 on the device).
    train = 1.4 * light_utils.generate_hemisphere_pts(
        "Fibonacci", sample_count=N_TRAIN, near_z=False).float().to(device)
    train_np = train.cpu().numpy()
    uz = held_np[:, 2].astype(np.float64) / np.linalg.norm(held_np.astype(np.float64), axis=1)
    bands_held = matup_lights.elevation_bands(held_np)
    # Sheet lights: the median-elevation grazing and mid held-out light, and the
    # HIGHEST high one - EVAL_LIGHT (index 0, 67 deg) already shows a high light.
    sheet = {}
    for b in BAND_NAMES:
        idx = sorted((int(i) for i in bands_held[b] if i != 0), key=lambda i: uz[i])
        if idx:
            sheet[b] = idx[-1] if b == "high" else idx[(len(idx) - 1) // 2]
    return {
        "held": torch.from_numpy(held_np).to(device), "held_np": held_np,
        "train": train, "train_np": train_np,
        # matup_render.render (:93): EVAL_LIGHT as a float32 (1, 3).
        "eval": torch.from_numpy(mr.EVAL_LIGHT).float().reshape(1, 3).to(device),
        "bands_held": bands_held,
        "bands_train": matup_lights.elevation_bands(train_np),
        "sheet": sheet, "uz_held": uz,
        "held_min_sep_deg": matup_lights.min_separation_deg(held_np, N_TRAIN),
        "train_equals_matup_lights": bool(np.array_equal(
            train_np, matup_lights.training_lights(N_TRAIN))),
    }


def read_manifests(cols, mats, expect_step):
    """sr_manifest.json of every arm x material; --expect-step fails fast here,
    before any rendering."""
    out = {}
    for c in cols:
        if not is_arm(c):
            continue
        per = {}
        for mat_dir in mats:
            name = os.path.basename(mat_dir.rstrip("/"))
            p = os.path.join(c["dir"], name, "sr_manifest.json")
            if not os.path.exists(p):
                continue
            with open(p) as fh:
                man = json.load(fh)
            per[name] = {k: man.get(k) for k in ("checkpoint", "step", "sub_name",
                                                 "matup_run", "scalar_collapse")}
            if expect_step is not None and man.get("step") is not None \
                    and int(man["step"]) != expect_step:
                die(2, "%s: arm %s was made from step %s, --expect-step is %d (%s)"
                    % (name, c["tag"], man["step"], expect_step, man.get("checkpoint")))
            # A tag that names a step ("render_r1_30k") must hold that step: the
            # head-to-head pairs and the per-step tables trust the suffix.
            if c["step"] and man.get("step") is not None \
                    and int(man["step"]) != 1000 * int(c["step"][:-1]):
                die(2, "%s: arm %s is named for %s but its sr_manifest step is %s (%s)"
                    % (name, c["tag"], c["step"], man["step"], man.get("checkpoint")))
            if man.get("scalar_collapse") not in (None, "mean"):
                print("  ! %s: arm %s collapsed grey maps with %r, not 'mean' - the render "
                      "loss trains on the channel mean" % (name, c["tag"], man["scalar_collapse"]))
        out[c["key"]] = per
        steps = sorted({v["step"] for v in per.values() if v.get("step") is not None})
        c["manifest_steps"] = steps
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/runs/matsynth")
    ap.add_argument("--arm", action="append", default=[],
                    help="TAG=DIR of an arm output folder (DIR/<mat>/{five maps}.png "
                         "+ sr_manifest.json); repeat per arm. TAG names the row through "
                         "variants/arms.json (+ _30k etc.); an unregistered tag shows as itself; "
                         "a _NNk suffix must match the manifest's step")
    ap.add_argument("--out", required=True, help="absolute output folder")
    ap.add_argument("--material", default="")
    ap.add_argument("--run", default="", help="MatUp output stamp; default is the newest")
    ap.add_argument("--chunk", type=int, default=10, help="lights rendered per call")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--expect-step", type=int, default=None,
                    help="every arm's sr_manifest.json step must equal this")
    args = ap.parse_args()

    out = check_out(args.out)
    if args.chunk < 1:
        die(2, "--chunk must be >= 1")
    arm_cols = parse_arms(args.arm)
    cols = ([{"key": "bilinear", "kind": "bilinear", "short": "bilinear"},
             {"key": "matup", "kind": "matup", "short": "matup"}] + arm_cols)
    names = {c["key"]: FIXED_NAMES.get(c["key"]) or display_name(c["tag"]) for c in cols}
    names["gt"] = FIXED_NAMES["gt"]

    mats = [os.path.join(args.root, args.material)] if args.material else \
        sorted(d for d in glob.glob(os.path.join(args.root, "*"))
               if os.path.isdir(os.path.join(d, "SVBRDF")))
    if not mats or not all(os.path.isdir(os.path.join(m, "GT")) for m in mats):
        die(2, "no material with GT/ under %s" % args.root)
    manifests = read_manifests(cols, mats, args.expect_step)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # matup_render.py:215-217: mandatory, and light falloff off as everywhere.
    Renderer.init_renderer(argparse.Namespace(light_falloff=False), device)
    lights = build_lights(device)
    os.makedirs(out, exist_ok=True)
    clear_stale(out, mats)
    print("[eval] device=%s materials=%d columns=%s chunk=%d out=%s" % (
        device, len(mats), ",".join(c["key"] for c in cols), args.chunk, out))
    print("[eval] lights train100=%d (== matup_lights: %s) heldout=%d sheet=%s" % (
        len(lights["train_np"]), lights["train_equals_matup_lights"], len(lights["held_np"]),
        {b: "#%d %.0fdeg" % (j, math.degrees(math.asin(lights["uz_held"][j])))
         for b, j in lights["sheet"].items()}))
    if not lights["train_equals_matup_lights"]:
        print("  ! MatUp's light table differs from matup_lights.training_lights - "
              "gate (a) (render_parity_torch.py --part lights) must be rerun")

    reports, overview = [], []
    with torch.no_grad():
        for d in mats:
            rep, sheet_rows, ov_cells = process(d, cols, args, device, lights)
            if rep is None:
                continue
            reports.append(rep)
            if not args.no_sheets:
                heads = ["EVAL_LIGHT (%.0f deg)" % math.degrees(math.asin(lights["uz_held"][0]))]
                for b in BAND_NAMES:
                    j = lights["sheet"].get(b)
                    heads.append("-" if j is None else "held-out #%d %s %.0f deg" % (
                        j, b, math.degrees(math.asin(lights["uz_held"][j]))))
                heads.append("|pred - GT| x%g (EVAL)" % HEAT_GAIN)
                rows = [(names[c["key"]], sheet_rows[c["key"]]) for c in cols
                        if c["key"] in sheet_rows] + [("GT", sheet_rows["gt"])]
                write_render_sheet(
                    os.path.join(out, "render_%s.png" % rep["material"]),
                    "%s  -  predicted SVBRDF rendered by MatUp's renderer    MatUp %s"
                    % (rep["material"], rep["run"]),
                    ["Raw maps, unguarded. Columns: EVAL_LIGHT, then one grazing / mid / high "
                     "held-out light (>= 2.5 deg from every training light direction).",
                     "|pred - GT|: RGB mean under EVAL_LIGHT x %g, FIXED for every row and "
                     "material (>= %.2f saturates); grey = GT-invalid texels, excluded from "
                     "every metric." % (HEAT_GAIN, 1.0 / HEAT_GAIN)],
                    heads, rows)
                overview.append((rep["material"], ov_cells))
    if not reports:
        die(2, "nothing scored")

    keys = [c["key"] for c in cols]
    if not args.no_sheets:
        write_overview(os.path.join(out, "overview_render_pred.png"), overview,
                       keys + ["gt"], names)

    common = common_materials(reports, keys)
    h2h = {p: head_to_head(common, *p) for p in head_to_head_pairs(cols)}
    steps = steps_tables(common, cols)
    tables, n_common = write_summary(os.path.join(out, "summary.md"), reports, cols, names,
                                     lights, h2h, steps)
    write_per_material(os.path.join(out, "per_material.md"), reports, cols, names)
    write_csv(os.path.join(out, "metrics_long.csv"), reports, cols, names)

    report = {
        "columns": [{"key": c["key"], "display": names[c["key"]], "kind": c["kind"],
                     "tag": c.get("tag"), "dir": c.get("dir"), "step": c.get("step"),
                     "manifest_steps": c.get("manifest_steps")} for c in cols],
        "manifests": manifests,
        "lights": {"train100": lights["train_np"], "heldout": lights["held_np"],
                   "train100_equals_matup_lights": lights["train_equals_matup_lights"],
                   "bands_train100": lights["bands_train"], "bands_heldout": lights["bands_held"],
                   "sheet_lights": lights["sheet"],
                   "heldout_min_sep_deg": lights["held_min_sep_deg"]},
        "constants": {"nz_min": NZ_MIN, "mask_min_norm": MASK_MIN_NORM,
                      "normal_norm_floor": NORMAL_NORM_FLOOR, "l21_eps": L21_EPS,
                      "self_check_tol": SELF_TOL, "heat_gain": HEAT_GAIN, "chunk": args.chunk,
                      "device": str(device)},
        "n_common": n_common,
        "means": tables,
        "head_to_head": {"%s_vs_%s" % p: v for p, v in h2h.items()},
        "steps": steps,
        "materials": reports,
    }
    with open(os.path.join(out, "report.json"), "w") as fh:
        json.dump(clean(report), fh, indent=1)
    print("[eval] %d material(s), %d common -> %s" % (
        len(reports), n_common, os.path.join(out, "summary.md")))


if __name__ == "__main__":
    main()
