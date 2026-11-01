#!/usr/bin/env python3
"""
Gate (a) and the torch half of gate (b): MatUp's own torch code against the TF
port of its renderer (scripts/deeptex/render_tf.py).

    docker compose run --rm matup python /workspace/variants/deeptex/render_loss/gates/render_parity_torch.py --part all

Runs in the matup container (torch 2.0, Python 3.10, no TensorFlow). The TF half,
scripts/deeptex/render_gate.py --part export, runs first in the deeptex container
(no torch) and leaves what it computed in parity_tf.npz, so the two stacks meet
only through that file. Nothing in external/matup_egsr is modified: its modules
are imported and driven from here, and bytecode writing is switched off before
the first import so no __pycache__ lands in the submodule.

WHAT EACH PART CAN AND CANNOT SEE
---------------------------------
lights -> parity_lights.json
    matup_lights.training_lights() against the literal main_optim.py:87-89
    expression `1.4 * generate_hemisphere_pts(...).float().to(device)`, on the
    CPU and on the GPU (MatUp scales on the device). The comparison is EXACT, bit
    for bit. The plausible mistake, scaling in float64 and casting afterwards,
    moves 58 of the 100 rows by one float32 ulp, so any tolerance would hide it.
    When parity_tf.npz exists, the table the deeptex container actually rendered
    with (numpy 1.23 there, torch here) is held to the same standard.

npz -> parity_torch.json, three checks:
  core      Every case's TF-DECODED inputs go through Renderer.torch_Render and
            render_utils.process_raw_render twice, in float32 and in float64 (the
            oracle). The tight bound is |TF - oracle| <= 2 |torch32 - oracle| +
            1e-6 per display value: the port about as far from the exact answer
            as MatUp's own float32 run. A fixed tolerance cannot work: at the
            roughness floor a = alpha^2 ~ 4e-6, and D amplifies float32 rounding
            in N.H by orders of magnitude, so correct code would fail near a
            highlight while wrong code passed elsewhere.
            The bound is STATISTICAL, not required of every value. In the graph,
            TF's float32 does not round in torch's order (Grappler turns
            x/sqrt(y) into x*rsqrt(y), CUDA rsqrtf is ~2 ulp; reductions may
            associate differently), and on the floor-roughness highlight shoulder
            those independent ulp errors reach a few 1e-6, where torch32's own
            error at that one value can be near 0 by luck. A numpy float32
            simulation of the synthetic texels under 2-4 ulp of rsqrt noise gave
            up to 5 / 17 / 107 values over the tight bound, each by <= 6.6e-6;
            torch's own op order gave 0. Pass, outside the ties: at most
            max(20, 1e-3 * values) values over the tight bound, and none over it
            by more than 2e-5. A wrong clamp or term moves display values by
            >= 1e-4 (typically >= 1e-3) and fails the second rule; a small error
            everywhere fails the first. The per-value count is reported.
            (texel, light) pairs whose raw N.H or L.H lies within 1e-6 of 0 are
            excluded, because a Heaviside sits there (renderer.py:129-130) and
            either side of it is right. Raw radiance is reported, not judged.
            This check is BLIND by construction to every decode, grid and
            light-table error, because torch is fed TF's own decode, positions
            and lights. The other two checks cover those.
  decode    The encoded stacks run through MatUp's own functions, then the same
            n_z guard, and the result is compared with TF's decode. The functions:
              - file_utils.srgb2linrgb
              - data_utils.normalize_np_norm (what read_img_as_float does to normal.png)
              - dataLoader.unpack_normal(t, True)
              - the load_set roughness floor
              - data_utils.get_renorm_clamped_maps
            Calling MatUp, not a replica, matters: a replica can share the very
            misconception it is meant to catch. Declared differences are excluded
            and counted rather than hidden:
              - the zero vector: normalize_np_norm leaves np.divide(where=)
                output uninitialised there (data_utils.py:50);
              - texels where TF's 1e-4 norm floor binds;
              - grey maps with unequal channels: TF averages, MatUp reads cv2
                channel 0 (file_utils.py:84-85);
              - heights outside [0, 1]: TF clips, position3d does not.
  anchored  Each side decodes the same eval GT PNGs on its own: TF from the
            bytes through the trainer's decode (render_gate.py), torch through
            matup_render.load_set + position3d. The comparison uses a 128^2 crop
            at the HR offset and all 100 lights. EVAL_LIGHT alone would not do:
            its x == y lets a consistent x/y transposition through.
            Then the full-material EVAL_LIGHT render is compared with two
            references, on GT-valid texels only (the n_z guard deliberately
            changes the others):
              - matup_render.render_source, unguarded: the function that wrote
                render_gt.png;
              - the committed render_gt.png itself.
            This check is what catches a y-flip, a shifted grid or a wrong light
            table.

NPZ LAYOUT this side reads (written by render_gate.py --part export). Keys are
'<case>/<field>'; '<case>__<field>' and '<case>_<field>' are accepted too, and
cases are discovered from their '<case>/display' key unless a `cases` array
lists them.
  lights               (L, 3) f32   the table TF rendered with (RenderLoss.lights_np)
  render_cfg           ()  str      optional: JSON of the validated render block
  <case>/albedo        (N, 3) f32   decode_maps output, flat over (B, H, W)
  <case>/metallic      (N, 1)
  <case>/normal        (N, 3)
  <case>/alpha         (N, 1)
  <case>/pos           (N, 3)
  <case>/valid         (N, 1)       optional (compared, informational)
  <case>/nz_guarded    (N, 1)       optional (compared, informational)
  <case>/radiance      (N, 3, L)    render_radiance (informational)
  <case>/display       (N, 3, L)    render_display
  <case>/enc           (B, 5, H, W, 3) or (N, 5, 3)  the stack decode_maps was given
  <case>/lights        (L, 3)       optional: overrides the global table
 anchored cases, i.e. any case with a material:
  <case>/material      ()  str      e.g. 0559_brick_uneven_stones
  <case>/offset        (2,) int     HR (y, x) of the crop in the 512 grid (default 192, 256)
  <case>/eval_display  (496, 496, 3) f32  full-512 EVAL_LIGHT render cropped [8:504, 8:504]
                                    ((512, 512, 3) or flat (N, 3, 1) is accepted as well)

Exit codes: 0 every requested part passed, 1 a check failed, 2 bad --part or an
--out under external/, 3 an input is missing (the json still says why).
"""
import argparse
import contextlib
import glob
import json
import os
import sys
import warnings

# Before the first import from external/: python would otherwise write
# __pycache__ into the submodule (matup_egsr's .gitignore covers *.pyc, but not
# the directory).
sys.dont_write_bytecode = True

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, REPO)
from variants.util import paths  # noqa: E402
EXTERNAL = paths.EXTERNAL
paths.use_matup()                                          # the `matup` package

from matup import data_utils, dataLoader, file_utils, light_utils, render_utils  # noqa: E402
from matup.renderer import Renderer  # noqa: E402

from variants.util import matup_lights  # noqa: E402
from variants.matup.baseline import matup_render  # noqa: E402

DEFAULT_NPZ = "/workspace/runs/deeptex/render_gate/parity_tf.npz"
DEFAULT_EVAL_ROOT = "/workspace/runs/matsynth"
DEFAULT_OUT = "/workspace/runs/deeptex/render_gate"
PARTS = ("lights", "npz")
OUT_NAMES = {"lights": "parity_lights.json", "npz": "parity_torch.json"}

# core: the tight bound |TF - oracle| <= CORE_FACTOR * |torch32 - oracle| + CORE_ABS
# is counted per value, not required of every value (module docstring). Outside
# Heaviside ties, at most max(CORE_BUDGET_MIN, CORE_BUDGET_FRAC * values) values
# may exceed it, and none by more than CORE_SLACK.
CORE_FACTOR = 2.0
CORE_ABS = 1e-6
CORE_BUDGET_FRAC = 1e-3
CORE_BUDGET_MIN = 20
CORE_SLACK = 2e-5
TIE_EPS = 1e-6                 # |raw N.H| or |raw L.H| below this: a Heaviside tie
DECODE_TOL = {"albedo": 1e-6, "metallic": 1e-6, "alpha": 1e-6, "z": 1e-6, "normal": 1e-5}
CROP_TOL = 5e-5                # anchored 128^2 crop, display space, valid texels
EVAL_TOL = 2.0 / 65535         # render_gt.png is 16-bit and written by truncation (file_utils.py:114-115)

FULL = 512
# matup_render.py:173 - border = tile_width // 2 with --tile-width 17 (:209), the
# rim MatUp never optimises. render_gt.png is stored already cropped by it.
BORDER = 17 // 2
ANCHOR_OFFSET = (192, 256)     # HR (y, x), used only if the npz does not say
CHUNK = 32768                  # texels per torch_Render call; shading is per texel

# render_tf.DEFAULTS for the keys read here. render_tf imports TensorFlow, which
# this image does not have, so the values are repeated; render_cfg in the npz
# overrides them.
TF_CFG = {"intensity": 8.0, "disp_val": 0.01, "min_roughness": 0.045, "min_alpha": 2e-3,
          "nz_min": 1e-3, "mask_min_norm": 0.1, "normal_norm_floor": 1e-4}

MAP_FIELDS = (("albedo", 3), ("metallic", 1), ("normal", 3), ("alpha", 1), ("pos", 3))
# Slot order of the encoded stack (render_tf.MAP_SLOTS == grouped_dataset.MAP_KEYS).
S_BC, S_NRM, S_RGH, S_MTL, S_HGT = 0, 1, 2, 3, 4

MATUP_EXPR = ('1.4 * light_utils.generate_hemisphere_pts("Fibonacci", sample_count=%d, '
              'near_z=False).float().to(device)')


class MissingInput(Exception):
    pass


def die(code, msg):
    print("[fatal] %s" % msg)
    sys.exit(code)


# ------------------------------------------------------------------- utilities

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if np.isfinite(f) else str(f)        # keep the file strict JSON
    return o


def write_json(path, obj):
    tmp = path + ".partial"
    with open(tmp, "w") as fh:
        json.dump(_jsonable(obj), fh, indent=2)
    os.replace(tmp, path)


def under(path, root):
    p = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    r = os.path.normcase(os.path.realpath(os.path.abspath(root))).rstrip(os.sep)
    return p == r or p.startswith(r + os.sep)


def absmax(a):
    a = np.asarray(a, np.float64)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a))) if np.all(np.isfinite(a)) else float("nan")


def _texts(a):
    return [x.decode() if isinstance(x, bytes) else str(x)
            for x in np.atleast_1d(np.asarray(a)).tolist()]


def flat(a, c):
    """(..., c) -> (N, c) float32. A (N,) array is accepted for c == 1."""
    a = np.asarray(a, np.float32)
    if c == 1 and a.ndim == 1:
        return a[:, None]
    if a.ndim < 2 or a.shape[-1] != c:
        raise ValueError("expected (..., %d), got %s" % (c, a.shape))
    return a.reshape(-1, c)


def as_render(a, n_l):
    """(..., 3, L) -> (N, 3, L) float64 (float32 values are exact in float64)."""
    a = np.asarray(a, np.float64)
    if a.ndim < 3 or a.shape[-2:] != (3, n_l):
        raise ValueError("render array has shape %s; expected (N, 3, %d)" % (a.shape, n_l))
    return a.reshape(-1, 3, n_l)


def enc_flat(enc, crop=None):
    """The encoded stack -> (N, 5, 3) in decode_maps' flattening order (B, H, W).

    crop = (oy, ox, size) cuts a full-material (B, 5, 512, 512, 3) stack to the
    exported crop when render_gate saved the stack before cropping.
    """
    e = np.asarray(enc, np.float32)
    if e.ndim == 4 and e.shape[0] == 5 and e.shape[-1] == 3:               # (5, H, W, 3)
        e = e[None]
    if e.ndim == 5 and e.shape[1] == 5 and e.shape[-1] == 3:               # (B, 5, H, W, 3)
        if crop and e.shape[2] != crop[2]:
            oy, ox, s = crop
            e = e[:, :, oy:oy + s, ox:ox + s]
        return np.transpose(e, (0, 2, 3, 1, 4)).reshape(-1, 5, 3)
    if e.ndim == 3 and e.shape[1:] == (5, 3):                              # (N, 5, 3)
        return e
    if e.ndim == 3 and e.shape[0] == 5 and e.shape[2] == 3:                # (5, N, 3)
        return np.transpose(e, (1, 0, 2))
    raise ValueError("enc has shape %s; expected (B, 5, H, W, 3) or (N, 5, 3)" % (e.shape,))


def eval_image(a):
    """An exported EVAL_LIGHT render -> (496, 496, 3) float64."""
    a = np.squeeze(np.asarray(a, np.float64))
    if a.ndim == 2 and a.shape[1] == 3:
        s = int(round(np.sqrt(a.shape[0])))
        if s * s != a.shape[0]:
            raise ValueError("eval_display has %d texels, not a square" % a.shape[0])
        a = a.reshape(s, s, 3)
    if a.ndim != 3 or a.shape[2] != 3 or a.shape[0] != a.shape[1]:
        raise ValueError("eval_display has shape %s; expected (496, 496, 3)" % (a.shape,))
    if a.shape[0] == FULL:
        a = a[BORDER:FULL - BORDER, BORDER:FULL - BORDER]
    if a.shape[0] != FULL - 2 * BORDER:
        raise ValueError("eval_display is %d^2; expected %d^2 (or %d^2 before the crop)"
                         % (a.shape[0], FULL - 2 * BORDER, FULL))
    return a


def read_png_rgb(path):
    """cv2 read, BGR -> RGB, scaled to [0, 1] float64. No normalisation."""
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise MissingInput("cannot read %s" % path)
    if a.ndim == 3:
        a = a[..., :3][..., ::-1]
    return a.astype(np.float64) / (65535.0 if a.dtype == np.uint16 else 255.0), a.dtype


def gt_valid(gt_dir, cfg):
    """The training mask on a GT normal PNG (render_tf.decode_maps `valid`):
    |2x - 1| >= mask_min_norm and unit n_z >= nz_min. Returns (valid, unit n_z)."""
    v, _ = read_png_rgb(os.path.join(gt_dir, "normal.png"))
    n = 2.0 * v - 1.0
    norm = np.linalg.norm(n, axis=-1)
    nz = n[..., 2] / np.where(norm > 0, norm, 1.0)
    return (norm >= cfg["mask_min_norm"]) & (nz >= cfg["nz_min"]), nz


# ------------------------------------------------------------------ npz access

class ParityNpz(object):
    """parity_tf.npz, resolved against the layout in the module docstring."""

    SEPS = ("/", "__", ".", "_")
    ALIASES = {
        "radiance": ("radiance", "tf_radiance", "raw"),
        "display": ("display", "tf_display"),
        "enc": ("enc", "encoded", "x5"),
        "offset": ("offset", "offset_hr", "crop_offset"),
    }
    REQUIRED = ("albedo", "metallic", "normal", "alpha", "pos", "display")

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.arrays = {k: z[k] for k in z.files}
        self.path = path
        self._sep = {}
        self.cases = self._discover()

    def _key(self, case, sep, field):
        for alias in self.ALIASES.get(field, (field,)):
            k = case + sep + alias
            if k in self.arrays:
                return k
        return None

    def _discover(self):
        listed = self.glob("cases", "case_names")
        candidates = []
        if listed is not None:
            candidates = [(c, s) for c in _texts(listed) for s in self.SEPS]
        else:
            for sep in self.SEPS:
                for alias in self.ALIASES["display"]:
                    suffix = sep + alias
                    candidates += [(k[:-len(suffix)], sep) for k in sorted(self.arrays)
                                   if k.endswith(suffix)]
        found, claimed = [], set()
        for case, sep in candidates:
            if not case or case in self._sep:
                continue
            keys = [self._key(case, sep, f) for f in self.REQUIRED]
            if None in keys or keys[-1] in claimed:
                continue
            claimed.update(keys)
            self._sep[case] = sep
            found.append(case)
        return found

    def glob(self, *names):
        for n in names:
            if n in self.arrays:
                return self.arrays[n]
        return None

    def get(self, case, field, required=True):
        k = self._key(case, self._sep[case], field)
        if k is None:
            if required:
                raise KeyError("npz has no %r for case %r (expected key %r)"
                               % (field, case, case + self._sep[case] + field))
            return None
        return self.arrays[k]

    def render_cfg(self):
        cfg = dict(TF_CFG)
        raw = self.glob("render_cfg", "rcfg")
        if raw is not None:
            cfg.update(json.loads(_texts(raw)[0]))
        return cfg

    def light_tables(self):
        out = {}
        if self.glob("lights") is not None:
            out["lights"] = self.glob("lights")
        for case in self.cases:
            k = self._key(case, self._sep[case], "lights")
            if k is not None:
                out[k] = self.arrays[k]
        return out

    def lights(self, case):
        a = self.get(case, "lights", required=False)
        if a is None:
            a = self.glob("lights")
        if a is None:
            raise KeyError("npz has no light table for case %r ('lights' or '%s%slights')"
                           % (case, case, self._sep[case]))
        return flat(a, 3)

    def material(self, case, eval_root):
        m = self.get(case, "material", required=False)
        if m is not None:
            return _texts(m)[0]
        pool = self.glob("anchored_materials", "materials")
        if pool is not None:
            pool = _texts(pool)
        elif os.path.isdir(eval_root):
            pool = sorted(d for d in os.listdir(eval_root)
                          if os.path.isdir(os.path.join(eval_root, d, "GT")))
        else:
            pool = []
        hits = [p for p in pool if p and p in case]
        return max(hits, key=len) if hits else None

    def offset(self, case):
        a = self.get(case, "offset", required=False)
        if a is not None:
            return tuple(int(v) for v in np.ravel(a)[:2]), "npz"
        a = self.get(case, "offset_lr", required=False)
        if a is not None:
            return tuple(4 * int(v) for v in np.ravel(a)[:2]), "npz offset_lr x4"
        a = self.glob("anchored_offset", "crop_offset", "offset")
        if a is not None:
            return tuple(int(v) for v in np.ravel(a)[:2]), "npz (global)"
        return ANCHOR_OFFSET, "default (not in npz)"

    def eval_display(self, case, material):
        a = self.get(case, "eval_display", required=False)
        if a is not None:
            return a
        for sep in self.SEPS:
            for k in (material + sep + "eval_display", "eval_display" + sep + material):
                if k in self.arrays:
                    return self.arrays[k]
        return None


# ---------------------------------------------------------------- MatUp's side

def matup_light_table(n, device):
    """main_optim.py:87-89, verbatim. `.float()` and `.to(device)` bind before
    `1.4 *`, so MatUp casts to float32 first and scales in float32 ON the device."""
    return 1.4 * light_utils.generate_hemisphere_pts(
        "Fibonacci", sample_count=n, near_z=False).float().to(device)


def compare_exact(ours, ref):
    """Bit-for-bit equality of two light tables, with the size of any difference."""
    ours, ref = np.asarray(ours), np.asarray(ref)
    res = {"shape": list(ours.shape), "ref_shape": list(ref.shape),
           "dtype": str(ours.dtype), "ref_dtype": str(ref.dtype)}
    if ours.dtype != np.float32 and ours.dtype.kind == "f":
        # A float64 copy of a float32 table is still the same table.
        cast = ours.astype(np.float32)
        if np.array_equal(cast.astype(ours.dtype), ours):
            res["note"] = "stored as %s; values are exact float32" % ours.dtype
            ours = cast
    if ours.shape != ref.shape or ours.dtype != np.float32 or ref.dtype != np.float32:
        res["equal"] = False
        return res
    bits = ours.view(np.int32) != ref.view(np.int32)
    res["equal"] = not bool(bits.any())
    res["values_differing"] = int(bits.sum())
    res["rows_differing"] = int(bits.any(axis=1).sum())
    res["max_abs"] = absmax(ours.astype(np.float64) - ref)
    res["max_ulp"] = int(np.abs(ours.view(np.int32).astype(np.int64)
                                - ref.view(np.int32).astype(np.int64)).max())
    return res


@contextlib.contextmanager
def renderer_dtype(dtype):
    """Run with Renderer's class tensors (renderer.py:11-12, cast and moved at
    :31-34) in `dtype`, then put MatUp's float32 state back - render_source reads
    z_vector through batch_render_tn (render_utils.py:51) afterwards."""
    saved = (Renderer.z_vector, Renderer.white_light)
    Renderer.z_vector = saved[0].to(dtype)
    Renderer.white_light = saved[1].to(dtype)
    try:
        yield
    finally:
        Renderer.z_vector, Renderer.white_light = saved


def torch_render(maps, lights, dtype, device, intensity):
    """Flat maps -> (radiance, display), each (N, 3, L) float64 numpy.

    Renderer.torch_Render (renderer.py:184-192) then process_raw_render
    (render_utils.py:134-137), chained as batch_render_tn chains them
    (render_utils.py:57-62) but without its reshape to an image. wo is
    Renderer.z_vector, as there (render_utils.py:51).
    """
    wo = Renderer.z_vector
    if wo.dtype != dtype:
        raise RuntimeError("Renderer.z_vector is %s, wanted %s: use renderer_dtype()"
                           % (wo.dtype, dtype))
    t = {k: torch.from_numpy(np.ascontiguousarray(maps[k], np.float32)).to(device=device, dtype=dtype)
         for k, _ in MAP_FIELDS}
    wi = torch.from_numpy(np.ascontiguousarray(lights, np.float32)).to(device=device, dtype=dtype)
    n, n_l = t["albedo"].shape[0], wi.shape[0]
    rads, disps = [np.zeros((0, 3, n_l))], [np.zeros((0, 3, n_l))]
    for s in range(0, n, CHUNK):
        e = s + CHUNK
        rad = Renderer.torch_Render(t["albedo"][s:e], t["metallic"][s:e], t["normal"][s:e],
                                    t["alpha"][s:e], t["pos"][s:e], wi, wo, intensity)
        disp = render_utils.process_raw_render(rad)
        rads.append(rad.cpu().numpy().astype(np.float64))
        disps.append(disp.cpu().numpy().astype(np.float64))
    return np.concatenate(rads), np.concatenate(disps)


def raw_dots(maps, lights, device):
    """float64 raw N.H and L.H, each (N, 1, L), before the clamp - only to find
    Heaviside ties. renderer.py:96-127 with MatUp's own torch_norm; run inside
    renderer_dtype(torch.float64)."""
    dt = torch.float64
    pos = torch.from_numpy(np.ascontiguousarray(maps["pos"])).to(device=device, dtype=dt)
    nrm = torch.from_numpy(np.ascontiguousarray(maps["normal"])).to(device=device, dtype=dt)
    wi = torch.from_numpy(np.ascontiguousarray(lights)).to(device=device, dtype=dt)
    wi = wi.t().unsqueeze(0)                                                    # :96
    wi_norm = data_utils.torch_norm(wi - pos.unsqueeze(-1), dimToNorm=1)        # :97, :102
    wo_norm = data_utils.torch_norm(Renderer.z_vector, dimToNorm=1).t().unsqueeze(0)  # :169
    normal = data_utils.torch_norm(nrm, dimToNorm=1).unsqueeze(-1)              # :170
    wh = wi_norm.unsqueeze(-1) + wo_norm.unsqueeze(-2)                          # :119
    n_, c_, i_, o_ = wh.shape
    wh = data_utils.torch_norm(wh.reshape((n_, c_, i_ * o_)), dimToNorm=1)      # :120-122
    n_dot_h = torch.sum(normal * wh, dim=1, keepdim=True)                       # :124
    l_dot_h = torch.sum(wi_norm.repeat(1, 1, o_) * wh, dim=1, keepdim=True)     # :127
    return n_dot_h.cpu().numpy(), l_dot_h.cpu().numpy()


def matup_decode(enc, cfg):
    """(N, 5, 3) encoded stack -> MatUp's decode of it, flat, plus masks.

    What matup_render.load_set (matup_render.py:53-76), position3d (:79-83) and
    render() (:90-92) do to a PNG folder, applied with MatUp's own functions to
    the values a PNG read would have produced. read_img_as_float returns float64
    (file_utils.py:54-62), so the arithmetic starts in float64 as it does there.
    Maps are laid out (1, N, C): get_renorm_clamped_maps exits - with status 0 -
    on 2-D maps (data_utils.py:85-86).
    """
    x = np.asarray(enc, np.float64)

    def row(a):
        return torch.from_numpy(np.ascontiguousarray(a)[None])

    albedo = file_utils.srgb2linrgb(row(x[:, S_BC])).float()                   # matup_render.py:62

    raw_norm = np.linalg.norm(2.0 * x[:, S_NRM] - 1.0, axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():   # the zero vector, excluded below
        warnings.simplefilter("ignore")
        unit_enc = data_utils.normalize_np_norm(x[:, S_NRM][None].copy())       # file_utils.py:87-88
    normal = dataLoader.unpack_normal(torch.from_numpy(unit_enc), True).float()  # matup_render.py:64-65

    # cv2 hands a 3-channel PNG over as BGR and read_img_as_float keeps index 0
    # of a grey map (file_utils.py:84-85), i.e. RGB channel 2.
    grey = {s: x[:, s, 2] for s in (S_RGH, S_MTL, S_HGT)}
    equal = {s: (x[:, s, 0] == x[:, s, 1]) & (x[:, s, 1] == x[:, s, 2]) for s in grey}
    rough = torch.clamp(row(grey[S_RGH]).float(), min=0.045)                    # matup_render.py:67-72
    alpha = torch.clamp(torch.pow(rough, 2.0),
                        min=Renderer.min_alpha, max=1.0).float()[..., None]     # :73-74
    metallic = row(grey[S_MTL]).float()[..., None]                              # :63
    z = (matup_render.DISP_VAL * (row(grey[S_HGT]).float() - 0.5)).float()      # :82

    albedo, metallic, normal, alpha = data_utils.get_renorm_clamped_maps(
        [albedo, metallic, normal, alpha])                                      # :92

    n_u = normal[0]
    nz = n_u[:, 2].numpy().astype(np.float64)
    guarded = nz < cfg["nz_min"]
    n_g = n_u.clone()
    n_g[:, 2] = torch.clamp(n_g[:, 2], min=cfg["nz_min"])                       # the guard (ours)
    n_g = data_utils.torch_norm(n_g, dimToNorm=1)
    return {"albedo": albedo[0].numpy(), "metallic": metallic[0].numpy(),
            "alpha": alpha[0].numpy(), "normal": n_g.numpy(), "normal_unguarded": n_u.numpy(),
            "z": z[0].numpy(), "height": grey[S_HGT], "raw_norm": raw_norm,
            "guarded": guarded,
            "valid": (raw_norm >= cfg["mask_min_norm"]) & (nz >= cfg["nz_min"]),
            "equal": equal}


# ------------------------------------------------------------------ part: lights

def part_lights(args, device):
    n = matup_lights.N_TRAIN
    ours = matup_lights.training_lights(n, matup_lights.RADIUS)
    report = {"expression": MATUP_EXPR % n, "ours": "matup_lights.training_lights(%d, %r)"
              % (n, matup_lights.RADIUS), "vs_matup": {}, "npz": {}}
    failures = []

    devices = [torch.device("cpu")] + ([device] if device.type != "cpu" else [])
    for d in devices:
        c = compare_exact(ours, matup_light_table(n, d).cpu().numpy())
        report["vs_matup"][str(d)] = c
        print("[lights] matup_lights vs main_optim.py:87-89 on %s: equal=%s rows_differing=%s"
              % (d, c["equal"], c.get("rows_differing")))
        if not c["equal"]:
            failures.append("matup_lights.training_lights differs from MatUp's table on %s" % d)

    if os.path.isfile(args.npz):
        tables = ParityNpz(args.npz).light_tables()
        report["npz"] = {"path": args.npz, "present": True, "tables": {}}
        if not tables:
            failures.append("%s holds no light table" % args.npz)
        for key, arr in sorted(tables.items()):
            a = np.asarray(arr).reshape(-1, 3)          # no cast: compare_exact judges the dtype
            c = compare_exact(a, matup_light_table(a.shape[0], torch.device("cpu")).numpy())
            report["npz"]["tables"][key] = c
            print("[lights] npz %r (built by the deeptex container) vs MatUp: equal=%s"
                  % (key, c["equal"]))
            if not c["equal"]:
                failures.append("npz light table %r differs from MatUp's" % key)
    else:
        report["npz"] = {"path": args.npz, "present": False,
                         "note": "not checked; --part npz checks it once render_gate.py "
                                 "--part export has written the file"}
        print("[lights] %s absent - the TF-side table is checked by --part npz" % args.npz)

    bands = matup_lights.elevation_bands(ours)
    report["bands"] = {k: len(v) for k, v in bands.items()}
    report["failures"] = failures
    report["pass"] = not failures
    return report


# --------------------------------------------------------------------- part: npz

def core_case(npz, case, lights, device, intensity):
    tf = {k: flat(npz.get(case, k), c) for k, c in MAP_FIELDS}
    n_l = lights.shape[0]
    n = tf["albedo"].shape[0]
    tf_disp = as_render(npz.get(case, "display"), n_l)
    rad = npz.get(case, "radiance", required=False)
    tf_rad = as_render(rad, n_l) if rad is not None else None
    for k, a in list(tf.items()) + [("display", tf_disp)]:
        if a.shape[0] != n:
            raise ValueError("%s has %d texels, albedo %d" % (k, a.shape[0], n))

    rad32, disp32 = torch_render(tf, lights, torch.float32, device, intensity)
    with renderer_dtype(torch.float64):
        rad64, disp64 = torch_render(tf, lights, torch.float64, device, intensity)
        n_dot_h, l_dot_h = raw_dots(tf, lights, device)

    tie = np.broadcast_to((np.abs(n_dot_h) < TIE_EPS) | (np.abs(l_dot_h) < TIE_EPS), tf_disp.shape)
    e_tf = np.abs(tf_disp - disp64)
    e_32 = np.abs(disp32 - disp64)
    bound = CORE_FACTOR * e_32 + CORE_ABS
    viol = ~(e_tf <= bound)                    # a NaN anywhere counts as a violation
    viol_out = viol & ~tie
    ex = np.where(viol_out, e_tf - bound, -np.inf)
    # a NaN outside the ties counts as an infinite excess
    excess_out = (float("inf") if np.isnan(ex).any() else float(ex.max())) if viol_out.any() else 0.0
    budget = max(CORE_BUDGET_MIN, int(np.ceil(CORE_BUDGET_FRAC * tf_disp.size)))
    off_tie = ~tie
    p999 = lambda a: float(np.percentile(a[off_tie], 99.9)) if off_tie.any() else 0.0   # noqa: E731

    res = {"texels": n, "lights": n_l, "values": int(tf_disp.size),
           "tie_pairs": int(tie[:, 0, :].sum()),
           "violations": int(viol.sum()), "violations_outside_ties": int(viol_out.sum()),
           "violation_budget": budget,
           "violation_frac": float(viol.mean()) if viol.size else 0.0,
           "max_excess": float(np.nanmax(np.where(viol, e_tf - bound, -np.inf))) if viol.any() else 0.0,
           "max_excess_outside_ties": excess_out, "slack": CORE_SLACK,
           "tf_err_max": absmax(e_tf), "torch32_err_max": absmax(e_32),
           "tf_err_p999_outside_ties": p999(e_tf), "torch32_err_p999_outside_ties": p999(e_32),
           "tf_vs_torch32_max": absmax(tf_disp - disp32),
           "nonfinite": {"tf": int((~np.isfinite(tf_disp)).sum()),
                         "torch32": int((~np.isfinite(disp32)).sum()),
                         "oracle": int((~np.isfinite(disp64)).sum())}}
    if tf_rad is not None:
        rel = lambda a: absmax(np.abs(a - rad64) / (1.0 + np.abs(rad64)))   # noqa: E731
        res["raw_rel_err_max"] = {"tf": rel(tf_rad), "torch32": rel(rad32)}
    if viol_out.any():
        score = np.where(viol_out, np.nan_to_num(e_tf - bound, nan=np.inf), -np.inf)
        i, c, j = np.unravel_index(int(np.argmax(score)), score.shape)
        res["worst"] = {"texel": int(i), "channel": int(c), "light": int(j),
                        "tf": tf_disp[i, c, j], "torch32": disp32[i, c, j], "oracle": disp64[i, c, j],
                        "raw_n_dot_h": float(n_dot_h[i, 0, j]), "raw_l_dot_h": float(l_dot_h[i, 0, j]),
                        "inputs": {k: tf[k][i].tolist() for k, _ in MAP_FIELDS},
                        "light_pos": lights[j].tolist()}
    # A non-finite TF value fails even on a tie pair; the exclusion is for sides
    # of a Heaviside, not for NaN.
    # Statistical rule (module docstring): few values over the tight bound, and
    # none of them far over it.
    res["pass"] = (res["violations_outside_ties"] <= budget and excess_out <= CORE_SLACK
                   and res["nonfinite"]["tf"] == 0 and res["nonfinite"]["oracle"] == 0)
    fails = [] if res["pass"] else [
        "core %s: %d display values outside 2|torch32-oracle|+1e-6 (budget %d), max excess %.3g "
        "(slack %.0e); non-finite tf %d oracle %d"
        % (case, res["violations_outside_ties"], budget, excess_out, CORE_SLACK,
           res["nonfinite"]["tf"], res["nonfinite"]["oracle"])]
    print("[core] %-34s texels=%-6d viol=%d (outside ties %d/%d, max excess %.3g, tie pairs %d) "
          "max|tf-o64|=%.3g max|t32-o64|=%.3g %s"
          % (case, n, res["violations"], res["violations_outside_ties"], budget, excess_out,
             res["tie_pairs"], res["tf_err_max"], res["torch32_err_max"],
             "PASS" if res["pass"] else "FAIL"))
    return res, fails


def decode_case(npz, case, cfg, crop, lights, device, intensity, synthetic):
    tf = {k: flat(npz.get(case, k), c) for k, c in MAP_FIELDS}
    n = tf["albedo"].shape[0]
    enc = npz.get(case, "enc", required=synthetic)
    if enc is None:
        # An anchored case's decode is checked against load_set in anchored_case.
        print("[decode] %-34s no enc - skipped (anchored: see decode_vs_load_set)" % case)
        return {"texels": n, "skipped": "no enc in npz", "pass": True}, []
    enc = enc_flat(enc, crop)
    if enc.shape[0] != n:
        raise ValueError("enc has %d texels, the decoded maps %d" % (enc.shape[0], n))
    m = matup_decode(enc, cfg)

    zero = m["raw_norm"] == 0
    floor = (m["raw_norm"] ** 2 < cfg["normal_norm_floor"]) & ~zero
    n_ok = ~(zero | floor)
    h = m["height"]
    h_ok = m["equal"][S_HGT] & (h >= 0.0) & (h <= 1.0)
    errs = {"albedo": absmax(tf["albedo"] - m["albedo"]),
            "normal": absmax((tf["normal"] - m["normal"])[n_ok]),
            "alpha": absmax((tf["alpha"] - m["alpha"])[m["equal"][S_RGH]]),
            "metallic": absmax((tf["metallic"] - m["metallic"])[m["equal"][S_MTL]]),
            "z": absmax((tf["pos"][:, 2] - m["z"])[h_ok])}
    res = {"texels": n, "max_abs": errs, "tol": DECODE_TOL,
           "excluded": {"normal_zero_vector": int(zero.sum()), "normal_norm_floor": int(floor.sum()),
                        "roughness_unequal_channels": int((~m["equal"][S_RGH]).sum()),
                        "metallic_unequal_channels": int((~m["equal"][S_MTL]).sum()),
                        "height_unequal_or_outside_0_1": int((~h_ok).sum())},
           "guarded_matup_side": int(m["guarded"].sum())}
    for key, mine in (("valid", m["valid"]), ("nz_guarded", m["guarded"])):
        a = npz.get(case, key, required=False)
        if a is not None:
            res[key + "_disagree"] = int(((flat(a, 1)[:, 0] > 0.5) != mine)[~zero].sum())

    if synthetic and n_ok.any():
        # Informational: MatUp's renderer on MatUp's own UNguarded normals - what
        # an n_z == 0 texel does without the guard (branchlessONB 0/0,
        # render_utils.py:96-97).
        mu = {"albedo": m["albedo"][n_ok], "metallic": m["metallic"][n_ok],
              "normal": m["normal_unguarded"][n_ok], "alpha": m["alpha"][n_ok],
              "pos": tf["pos"][n_ok]}
        _, disp = torch_render(mu, lights, torch.float32, device, intensity)
        res["unguarded_nonfinite_texels"] = int((~np.isfinite(disp)).any(axis=(1, 2)).sum())

    bad = [k for k, v in errs.items() if not v <= DECODE_TOL[k]]
    res["pass"] = not bad
    fails = ["decode %s: %s" % (case, ", ".join("%s %.3g > %.0e" % (k, errs[k], DECODE_TOL[k])
                                                for k in bad))] if bad else []
    print("[decode] %-34s %s %s" % (case, " ".join("%s=%.2g" % kv for kv in sorted(errs.items())),
                                   "PASS" if res["pass"] else "FAIL"))
    return res, fails


def committed_render(mat_dir, material):
    """render_gt.png as matup_render.process wrote it: into the newest run folder
    (matup_render.py:165-181). The newest folder that holds one is used."""
    runs = sorted(d for d in glob.glob(os.path.join(mat_dir, "matup-output-%s" % material, "*"))
                  if os.path.isfile(os.path.join(d, "render_gt.png")))
    return os.path.join(runs[-1], "render_gt.png") if runs else None


def anchored_case(npz, case, material, args, device, cfg, intensity):
    fails = []
    mat_dir = os.path.join(args.eval_root, material)
    gt_dir = os.path.join(mat_dir, "GT")
    if not os.path.isdir(gt_dir):
        raise MissingInput("no %s (anchored case %r)" % (gt_dir, case))
    (oy, ox), off_src = npz.offset(case)
    n_l = npz.lights(case).shape[0]
    tf_disp = as_render(npz.get(case, "display"), n_l)
    tf = {k: flat(npz.get(case, k), c) for k, c in MAP_FIELDS}
    n = tf_disp.shape[0]
    s = int(round(np.sqrt(n)))
    if s * s != n or oy < 0 or ox < 0 or oy + s > FULL or ox + s > FULL:
        raise ValueError("anchored %s: %d texels at offset (%d, %d) is not one square crop of "
                         "the %d grid" % (case, n, oy, ox, FULL))
    res = {"material": material, "offset_hr_yx": [oy, ox], "offset_source": off_src, "size": s}

    # --- torch side, decoded from the PNGs by MatUp's own loader
    albedo, metallic, normal, alpha = matup_render.load_set(gt_dir, "", True, True)   # MatSynth: OpenGL
    h = torch.from_numpy(file_utils.read_img_as_float(
        os.path.join(gt_dir, "height.png"), False)).float()                           # :101-103
    pos = matup_render.position3d(h, albedo.shape[0])                                 # full grid, then crop
    if albedo.shape[0] != FULL:
        raise ValueError("%s GT is %d^2, not %d^2" % (material, albedo.shape[0], FULL))
    cut = lambda t: t[oy:oy + s, ox:ox + s]                                           # noqa: E731
    maps = data_utils.get_renorm_clamped_maps(
        [cut(albedo), cut(metallic), cut(normal), cut(alpha), cut(pos)])              # render() :92
    mu = {k: maps[i].reshape(-1, maps[i].shape[-1]).numpy() for i, (k, _) in enumerate(MAP_FIELDS)}

    valid_png, _ = gt_valid(gt_dir, cfg)
    valid_m = valid_png[oy:oy + s, ox:ox + s].reshape(-1)
    guarded = mu["normal"][:, 2] < cfg["nz_min"]
    if guarded.any():
        g = torch.from_numpy(mu["normal"].copy())
        g[:, 2] = torch.clamp(g[:, 2], min=cfg["nz_min"])
        mu["normal"] = data_utils.torch_norm(g, dimToNorm=1).numpy()
    tf_valid = npz.get(case, "valid", required=False)
    tf_valid = (flat(tf_valid, 1)[:, 0] > 0.5) if tf_valid is not None else np.ones(n, bool)
    mask = valid_m & tf_valid & ~guarded

    # decode, side by side (pinpoints which map a render mismatch comes from)
    dec = {"albedo": absmax(tf["albedo"] - mu["albedo"]),
           "metallic": absmax(tf["metallic"] - mu["metallic"]),
           "alpha": absmax(tf["alpha"] - mu["alpha"]),
           "normal": absmax((tf["normal"] - mu["normal"])[mask]),
           "z": absmax(tf["pos"][:, 2] - mu["pos"][:, 2])}
    xy_exact = bool(np.array_equal(tf["pos"][:, :2], mu["pos"][:, :2]))
    bad = [k for k, v in dec.items() if not v <= DECODE_TOL[k]]
    res["decode_vs_load_set"] = {"max_abs": dec, "tol": DECODE_TOL, "pos_xy_exact": xy_exact,
                                 "pass": not bad and xy_exact}
    if bad or not xy_exact:
        fails.append("anchored %s decode vs load_set: %s%s" % (
            material, ", ".join("%s %.3g" % (k, dec[k]) for k in bad),
            "" if xy_exact else " pos xy not identical to compute_2D_position_grid"))

    # --- 128^2 crop, all lights, MatUp's own light table
    lights_m = matup_light_table(n_l, device).cpu().numpy()
    _, t_disp = torch_render(mu, lights_m, torch.float32, device, intensity)
    d = np.abs(tf_disp - t_disp)
    dv, di = d[mask], d[~mask]
    crop = {"texels": n, "valid": int(mask.sum()), "tf_invalid": int((~tf_valid).sum()),
            "matup_invalid": int((~valid_m).sum()), "guarded_torch_side": int(guarded.sum()),
            "max_abs_valid": absmax(dv), "mean_abs_valid": float(dv.mean()) if dv.size else 0.0,
            "p999_abs_valid": float(np.percentile(dv, 99.9)) if dv.size else 0.0,
            "max_abs_invalid": absmax(di), "tol": CROP_TOL}
    crop["pass"] = bool(mask.any()) and crop["max_abs_valid"] <= CROP_TOL
    res["crop"] = crop
    if not crop["pass"]:
        fails.append("anchored %s crop: max|tf - torch| %.3g on %d valid texels (tol %.0e)"
                     % (material, crop["max_abs_valid"], crop["valid"], CROP_TOL))
    print("[anchored] %-26s crop %d^2 @(%d,%d) max|tf-torch|=%.3g on %d/%d valid, decode %s %s"
          % (material, s, oy, ox, crop["max_abs_valid"], crop["valid"], n,
             "ok" if res["decode_vs_load_set"]["pass"] else "BAD", "PASS" if crop["pass"] else "FAIL"))

    # --- full material under EVAL_LIGHT
    ev = {"tol": EVAL_TOL}
    raw = npz.eval_display(case, material)
    if raw is None:
        fails.append("anchored %s: npz has no eval_display (expected %r)"
                     % (material, case + npz._sep[case] + "eval_display"))
        ev["pass"] = False
        res["eval_light"] = ev
        return res, fails
    tf_ev = eval_image(raw)
    vmask = valid_png[BORDER:FULL - BORDER, BORDER:FULL - BORDER]
    # the function that wrote render_gt.png (matup_render.py:97-111), unguarded
    fresh = matup_render.render_source(mat_dir, "GT", device, BORDER).cpu().numpy().astype(np.float64)
    com_path = committed_render(mat_dir, material)
    if com_path is None:
        raise MissingInput("no committed render_gt.png under %s/matup-output-%s/*"
                           % (mat_dir, material))
    com, com_dtype = read_png_rgb(com_path)

    def cmp(a, b):
        if a.shape != b.shape:
            return {"shape_mismatch": [list(a.shape), list(b.shape)], "pass": False}
        dd = np.abs(a - b)
        out = {"max_abs_valid": absmax(dd[vmask]), "max_abs_invalid": absmax(dd[~vmask]),
               "over_tol_valid": int((dd[vmask] > EVAL_TOL).sum())}
        out["pass"] = out["max_abs_valid"] <= EVAL_TOL
        return out

    ev.update({"valid": int(vmask.sum()), "invalid": int((~vmask).sum()),
               "committed": com_path, "committed_dtype": str(com_dtype),
               "tf_vs_render_source": cmp(tf_ev, fresh),
               "tf_vs_committed": cmp(tf_ev, com),
               "render_source_vs_committed": cmp(fresh, com)})          # informational
    if tf_ev.shape == com.shape:
        q = np.floor(65535.0 * np.clip(tf_ev, 0.0, 1.0)) / 65535.0     # file_utils.py:114-115
        ev["tf_truncated_equals_committed_frac"] = float((q == com)[vmask].mean())
    if com_dtype != np.uint16:
        fails.append("anchored %s: %s is %s, expected 16-bit" % (material, com_path, com_dtype))
    ev["pass"] = (ev["tf_vs_render_source"]["pass"] and ev["tf_vs_committed"]["pass"]
                  and com_dtype == np.uint16)
    for k in ("tf_vs_render_source", "tf_vs_committed"):
        if not ev[k]["pass"]:
            fails.append("anchored %s eval light %s: %s" % (material, k, ev[k]))
    res["eval_light"] = ev
    print("[anchored] %-26s eval light: max|tf-render_source|=%.3g max|tf-committed|=%.3g "
          "(tol %.3g, %d valid) %s" % (material, ev["tf_vs_render_source"].get("max_abs_valid", -1),
                                       ev["tf_vs_committed"].get("max_abs_valid", -1), EVAL_TOL,
                                       ev["valid"], "PASS" if ev["pass"] else "FAIL"))
    return res, fails


def part_npz(args, device):
    if not os.path.isfile(args.npz):
        raise MissingInput("no %s - run render_gate.py --part export in the deeptex container first"
                           % args.npz)
    npz = ParityNpz(args.npz)
    cfg = npz.render_cfg()
    intensity = float(cfg["intensity"])
    # matup_render.py:215-217: init_renderer only reads .light_falloff, but
    # torch_render_base asserts is_init and z_vector has the wrong shape without it.
    Renderer.init_renderer(argparse.Namespace(light_falloff=False), device)

    report = {"npz": args.npz, "eval_root": args.eval_root, "device": str(device),
              "torch": torch.__version__, "render_cfg": cfg, "cases": list(npz.cases),
              "thresholds": {"core": "tight bound |tf-oracle64| <= %g*|torch32-oracle64| + %g; "
                                     "outside Heaviside ties (|raw dot| < %g) at most "
                                     "max(%d, %g*values) values over it, none by more than %g"
                                     % (CORE_FACTOR, CORE_ABS, TIE_EPS, CORE_BUDGET_MIN,
                                        CORE_BUDGET_FRAC, CORE_SLACK),
                             "decode": DECODE_TOL, "anchored_crop": CROP_TOL,
                             "anchored_eval_light": EVAL_TOL}}
    failures = []
    for k, mv in (("intensity", 8.0), ("disp_val", 0.01), ("min_roughness", 0.045),
                  ("min_alpha", 2e-3)):
        if float(cfg[k]) != mv:
            failures.append("render_cfg %s = %r, MatUp uses %r" % (k, cfg[k], mv))

    tables = npz.light_tables()
    report["lights"] = {}
    for key, arr in sorted(tables.items()):
        a = np.asarray(arr).reshape(-1, 3)              # no cast: compare_exact judges the dtype
        c = compare_exact(a, matup_light_table(a.shape[0], torch.device("cpu")).numpy())
        report["lights"][key] = c
        if not c["equal"]:
            failures.append("npz light table %r differs from main_optim.py:87-89" % key)
    if not tables:
        failures.append("npz holds no light table ('lights' or '<case>/lights')")

    if not npz.cases:
        failures.append("no cases in %s (need '<case>/display' + decoded maps); keys: %s"
                        % (args.npz, sorted(npz.arrays)))
    kind = {c: npz.material(c, args.eval_root) for c in npz.cases}
    anchored = [c for c in npz.cases if kind[c]]
    synthetic = [c for c in npz.cases if not kind[c]]
    report["anchored_cases"] = {c: kind[c] for c in anchored}
    report["synthetic_cases"] = synthetic
    if npz.cases and not anchored:
        failures.append("no anchored case (one with a 'material' naming an eval material)")
    if npz.cases and not synthetic:
        failures.append("no synthetic case")
    print("[npz] %s: %d case(s) - anchored %s, synthetic %s"
          % (args.npz, len(npz.cases), [kind[c] for c in anchored], synthetic))

    report["core"], report["decode"], report["anchored"] = {}, {}, {}
    for case in npz.cases:
        try:
            res, fl = core_case(npz, case, npz.lights(case), device, intensity)
        except (KeyError, ValueError) as e:
            res, fl = {"pass": False, "error": str(e)}, ["core %s: %s" % (case, e)]
        report["core"][case] = res
        failures += fl

    for case in npz.cases:
        crop = None
        if kind[case]:
            (oy, ox), _ = npz.offset(case)
            crop = (oy, ox, int(round(np.sqrt(flat(npz.get(case, "albedo"), 3).shape[0]))))
        try:
            res, fl = decode_case(npz, case, cfg, crop, npz.lights(case), device, intensity,
                                  synthetic=not kind[case])
        except (KeyError, ValueError) as e:
            res, fl = {"pass": False, "error": str(e)}, ["decode %s: %s" % (case, e)]
        report["decode"][case] = res
        failures += fl

    for case in anchored:
        try:
            res, fl = anchored_case(npz, case, kind[case], args, device, cfg, intensity)
        except (KeyError, ValueError) as e:
            res, fl = {"pass": False, "error": str(e)}, ["anchored %s: %s" % (case, e)]
        report["anchored"][case] = res
        failures += fl

    report["failures"] = failures
    report["pass"] = not failures
    for f in failures:
        print("  ! %s" % f)
    return report


# ------------------------------------------------------------------------- main

def parse_parts(text):
    names = [p.strip() for p in text.split(",") if p.strip()]
    if names == ["all"]:
        return list(PARTS)
    bad = [p for p in names if p not in PARTS]
    if bad or not names:
        die(2, "--part takes a comma list of %s, or all; got %r" % (",".join(PARTS), text))
    return [p for p in PARTS if p in names]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all", help="lights,npz | all (default all)")
    ap.add_argument("--npz", default=DEFAULT_NPZ,
                    help="render_gate.py --part export output (default %(default)s)")
    ap.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT,
                    help="eval materials, <root>/<mat>/GT and matup-output-<mat>/ (default %(default)s)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="json folder (default %(default)s)")
    args = ap.parse_args()

    parts = parse_parts(args.part)
    # The container's working dir IS the submodule, so a relative --out lands
    # in it too; both spellings of external/ are refused.
    if under(args.out, EXTERNAL) or under(args.out, "/workspace/external"):
        die(2, "--out %s is under external/ - write to /workspace/runs/..." % args.out)
    os.makedirs(args.out, exist_ok=True)
    for p in parts:                       # never leave a stale PASS behind a crash
        stale = os.path.join(args.out, OUT_NAMES[p])
        if os.path.exists(stale):
            os.remove(stale)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[*] torch %s on %s; parts %s" % (torch.__version__, device, ",".join(parts)))
    if device.type == "cpu":
        print("[warn] no GPU visible: torch32 runs on the CPU, so the core bound is CPU float32 "
              "error while the TF side ran on the GPU")

    verdict = {}
    with torch.no_grad():
        for p in parts:
            path = os.path.join(args.out, OUT_NAMES[p])
            try:
                report = part_lights(args, device) if p == "lights" else part_npz(args, device)
            except MissingInput as e:
                write_json(path, {"pass": False, "error": str(e)})
                die(3, str(e))
            write_json(path, report)
            verdict[p] = report["pass"]
            print("[parity] %s: %s -> %s" % (p, "PASS" if report["pass"] else "FAIL", path))
    sys.exit(0 if all(verdict.values()) else 1)


if __name__ == "__main__":
    main()
