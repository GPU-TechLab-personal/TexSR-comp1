#!/usr/bin/env python3
"""Render-loss gates (b) and (d), TensorFlow side: the parity export and six checks.

Runs in the deeptex container (TF 2.10.1, Python 3.8), ON THE GPU:

    docker compose run --rm deeptex python /workspace/scripts/deeptex/render_gate.py --part export
    docker compose run --rm matup   python /workspace/scripts/matup/render_parity_torch.py --part all
    docker compose run --rm deeptex python /workspace/scripts/deeptex/render_gate.py \\
        --part grads,flow,grid,chunks,mask,graph

(run_deeptex.ps1 -Stage gate-render -Gate parity runs exactly that sequence.)

Each part writes <out>/<part>.json holding "pass", the failed check names and the
numbers behind every check, and deletes its stale JSON before it starts so a
crash can never leave an old PASS behind. Exit status: 0 all requested parts
passed, 1 any failed, 2 bad argument or output path, 6 no GPU visible.

Only the `render:` block of --cfg is read (plus gt_size/scale, so the crops here
are the trainer's crops). Nothing is trained, nothing under external/ is
imported, and every output goes to an absolute path outside the submodules.

WHY ON THE GPU AND IN GRAPH MODE. The numbers have to describe the path the
trainer runs: render_tf inside a tf.function, placed on the RTX card. Two things
differ off that path. TF 2.10 runs MatMul/Einsum in TF32 on Ampere (render_tf.py
module docstring), and Grappler may rewrite graph arithmetic (x/sqrt(y) ->
x*rsqrt(y)) that eager mode never sees. So a missing GPU is fatal, and every
render here goes through tf.function.

THE PARTS

  export  writes <out>/parity_tf.npz for scripts/matup/render_parity_torch.py,
          which holds the TF port to MatUp's own torch code (gate b). Two kinds
          of case:
            synthetic  600 edge texels: every clip boundary, the roughness floor,
                       normals with n_z = 1, 0.5, 0 EXACTLY and -0.3, the zero
                       vector, a too-short, a too-long and an out-of-[0,1] normal.
                       These are where a port goes subtly wrong.
            anchored   0559_brick_uneven_stones and th_brick_wall_005 (min n_z
                       0.21, so the guard is idle; 0844 is NOT used because the
                       guard moves 101 of its texels): GT PNG bytes decoded by the
                       training decoder (tf.io.read_file + decode_png(channels=3)),
                       a 128^2 crop at HR offset (192, 256) cut together with the
                       full 512^2 MatUp grid, rendered under all 100 training
                       lights. The torch side decodes the SAME PNGs independently
                       with matup_render.load_set, which is the only way a decode,
                       y-flip or grid error can show up: a parity test fed with
                       TF-decoded inputs is blind to all three by construction.
          Per anchored material it also renders the full 512^2 material under
          EVAL_LIGHT, cropped [8:504] like the committed render_gt.png
          (matup_render.py:173,176-181; border = tile_width 17 // 2), and compares
          against that file right here, informationally (the torch script owns
          the verdict). EVAL_LIGHT is symmetric in x and y (0.276172, 0.276172),
          so that check cannot see an x/y transposition; the 100-light crop can.
  grads   the synthetic edge texels as a tf.Variable, l2,1 against a fixed
          target: loss and every gradient finite, ||g|| < 1e6, with light_chunks
          1 and 4. A NaN here would reach the generator through apply_gradients
          before the trainer's loss check could see it. Also an UNCLIPPED case,
          GT + N(0, 0.3), reporting the fraction of entries whose render gradient
          is exactly zero (the clip/floor dead zones of an unbounded conv_last,
          modules/models.py:109) - informational.
  flow    two training materials from the PNG pairs, one with textured metallic:
          l2,1(GT, GT) through RenderLoss.probe equals float32(sqrt(1e-15)); with
          SR = clip(GT + N(0, 0.05)) the gradient reaches baseColor, normal,
          roughness and metallic, height gets <= 1e-2 of the median of baseColor/
          normal/roughness (texel z spans 0.01 while the lowest light sits at
          z = 0.099, so the render barely sees height - declared, not fixed), and
          a 1e-3 step along -g/||g|| lowers the loss.
  grid    for all 8 (flip, rot90 k): render(aug(maps), aug(pos)) == aug(render(
          maps, pos)) within 1e-6, the crops equal an independent numpy flip-then-
          rot90 of the identity crop exactly, and the identity pos patch equals
          position_grid_np(512)[4oy:4oy+128, 4ox:4ox+128] exactly. Shading is
          strictly per texel with a fixed (0,0,1) view, so this is an exact
          permutation; it pins the ORDER of the tf.image ops and the grid cut.
  chunks  light_chunks = 4 (tf.recompute_grad) against 1 on the training batch
          shape, under a persistent tape inside a tf.function as in the trainer:
          loss rel diff <= 1e-6, max|g4 - g1| / max|g1| <= 1e-4, and the peak GPU
          memory of each (informational - whether recompute_grad saves memory in
          TF 2.10 is exactly what is unknown).
  mask    on materials with no invalid GT texel the masked l2,1 equals the
          unmasked one (rel 1e-6); on materials with invalid texels the masked
          value equals sum(valid * e) / (max(sum valid, 1) * L) recomputed in
          float64, and the invalid fraction is reported.
  graph   the concrete functions of RenderLoss.target, .loss and loss+gradient
          (chunks 1 and 4) and the eval-light render contain no MatMul /
          BatchMatMul* / Einsum / _FusedMatMul op, nested functions included; and
          render_tf.MAP_SLOTS == grouped_dataset.MAP_KEYS.

parity_tf.npz LAYOUT (np.load(..., allow_pickle=False) works)
  manifest            0-d str: JSON with everything below plus versions and cfg
  render_cfg          0-d str: JSON of the validated render: block (RenderLoss.cfg)
  cases               (C,) str: "synthetic", "anchored_<material>", ...
  lights              (L, 3) float32: RenderLoss.lights, the table every case used
  eval_light          (1, 3) float32: matup_lights.EVAL_LIGHT
  map_keys            (5,) str: slot order of every encoded stack
  anchored_materials  (A,) str
  Per case c (N = B*H*W texels; flat index n = (b*H + h)*W + w, as decode_maps):
  c__encoded          (B, 5, H, W, 3) float32  network/encoded space, the decode input
  c__encoded_flat     (5, N, 3) float32        the same, slot-major in the N order
  c__pos_xy           (B, H, W, 2) float32     XY grid fed to decode_maps
  c__albedo (N,3) c__metallic (N,1) c__normal (N,3) c__alpha (N,1) c__pos (N,3)
  c__valid (N,1) c__nz_guarded (N,1)           TF-decoded renderer inputs (float32)
  c__radiance         (N, 3, L) float32        linear, before Reinhard + gamma
  c__display          (N, 3, L) float32        process_raw_render(radiance)
  synthetic only:     c__normal_kind (N,) str, c__zero_vector (N,) bool,
                      c__albedo_enc (N,3), c__rough_enc (N,), c__metal_enc (N,),
                      c__height_enc (N,) (encoded values; heights -0.2/1.2 are
                      CLIPPED to [0,1] by decode_maps - position3d never clips)
  anchored only:      c__material 0-d str, c__offset (2,) int32 HR (y, x) in the
                      512 grid, c__bit_depth (5,) int32 per slot, c__eval_display
                      (496, 496, 3) float32, c__eval_valid / c__eval_nz_guarded
                      (496, 496) float32, c__eval_border 0-d int32 (8)
  export.json records the npz sha256, so a parity_torch.json can be tied to the
  export it read.
"""

import argparse
import collections
import datetime
import glob
import hashlib
import json
import os
import sys
import time
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                               # render_tf, grouped_dataset
sys.path.insert(0, os.path.join(HERE, os.pardir, "common"))            # matup_lights

import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402

import grouped_dataset  # noqa: E402
import matup_lights  # noqa: E402
import render_tf  # noqa: E402

PARTS = ("export", "grads", "flow", "grid", "chunks", "mask", "graph")
MAP_KEYS = grouped_dataset.MAP_KEYS
BC, NRM, RGH, MTL, HGT = range(5)

DEFAULT_CFG = "/workspace/configs/deeptex/esrgan_matsynth_render_r1.yaml"
DEFAULT_OUT = "/workspace/runs/deeptex/render_gate"
EXTERNAL = ("/workspace/external",
            os.path.normpath(os.path.join(HERE, os.pardir, os.pardir, "external")))

# Gate (b) anchors: both have min GT n_z 0.21, so the guard never fires on them.
ANCHORED = ("0559_brick_uneven_stones", "th_brick_wall_005")
HR_OFFSET = (192, 256)
# matup_render.py:173,209: b = tile_width // 2 with tile_width 17; render_gt.png
# is the 512 render cropped by b on every side.
EVAL_BORDER = 8
# Grid gate LR offsets: the anchored crop (x4 = 192, 256) and the overfit gate's
# --overfit-fixed default.
GRID_OFFSETS_LR = ((48, 64), (16, 24))
# Mask gate: ~13% invalid texels in the (192, 256) crop, and 100% (a constant
# RGB that is not a normal map).
MASK_INVALID = ("1162_wood_planks_aged", "1652_fur_cow_long")
FLOW_SCAN_MAX = 300
METAL_FRAC = (0.25, 0.75)

# Tolerances, fixed before running (spec section 8).
FLOOR_REL = 1e-4
HEIGHT_RATIO_MAX = 1e-2
DESCENT_STEP = 1e-3
GRID_TOL = 1e-6
CHUNK_LOSS_REL = 1e-6
CHUNK_GRAD_REL = 1e-4
MASK_REL = 1e-6
MASK_FORMULA_REL = 1e-4
GRAD_NORM_MAX = 1e6
EXPORT_TRAIN_PATH_MAX = 1e-4
# Reinhard + gamma maps [0, inf) into [0, 1); a saturated highlight lands on
# 1.055 * 1 - 0.055, which float32 rounding may put one ulp either side of 1.
DISPLAY_MAX_SLACK = 1e-6
NOISE_FLOW = 0.05
NOISE_UNCLIPPED = 0.3
FORBIDDEN_OPS = frozenset(["MatMul", "BatchMatMul", "BatchMatMulV2", "BatchMatMulV3",
                           "Einsum", "_FusedMatMul"])

# Synthetic edge texels (encoded / network space).
ALBEDO_ENC = (-0.2, 0.0, 0.5, 1.0, 1.3)
ROUGH_ENC = (0.0, 0.02, 0.045, 0.5, 1.0, 1.2)
METAL_ENC = (-0.1, 0.0, 0.5, 1.0, 1.1)
HEIGHT_ENC = (-0.2, 0.0, 0.5, 1.0, 1.2)
AZIMUTHS = (0, 45, 90, 180, 270)


# --------------------------------------------------------------------- plumbing

def die(code, msg):
    print("[fatal] %s" % msg, flush=True)
    sys.exit(code)


def refuse_external(path, what):
    """The containers start INSIDE the submodules (docker/compose.yaml
    working_dir), so a relative path would land in external/ and dirty it."""
    p = os.path.abspath(path)
    for root in EXTERNAL:
        if p == root or p.startswith(root.rstrip("/\\") + os.sep) or p.startswith(root + "/"):
            die(2, "%s %s is under %s - outputs must go to /workspace/runs/..." % (what, p, root))
    return p


def _clean(o):
    """JSON-safe copy: numpy/TF values to Python, non-finite floats to strings.
    Strict JSON on purpose - run_deeptex.ps1 reads these files with
    ConvertFrom-Json, which rejects NaN and Infinity tokens."""
    if isinstance(o, dict):
        return collections.OrderedDict((str(k), _clean(v)) for k, v in o.items())
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (tf.Tensor, tf.Variable)):
        o = o.numpy()
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if np.isfinite(f) else str(f)
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    return o


def write_json(path, obj):
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=1, allow_nan=False)
    os.replace(tmp, path)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fmt(numbers):
    out = []
    for k, v in sorted(numbers.items()):
        if isinstance(v, float):
            out.append("%s=%.4g" % (k, v))
        elif isinstance(v, (list, tuple, dict)):
            continue
        else:
            out.append("%s=%s" % (k, v))
    return " ".join(out)


class Part(object):
    """Collects named checks (each pass/fail plus its numbers) for one JSON."""

    def __init__(self, name):
        self.name = name
        self.checks = collections.OrderedDict()
        self.info = collections.OrderedDict()
        self.t0 = time.time()

    def check(self, key, ok, **numbers):
        ok = bool(ok)
        entry = collections.OrderedDict([("pass", ok)])
        for k in sorted(numbers):
            entry[k] = numbers[k]
        self.checks[key] = entry
        print("[%s] %s %s %s" % (self.name, "ok  " if ok else "FAIL", key, _fmt(numbers)),
              flush=True)
        return ok

    def warn(self, msg):
        self.info.setdefault("warnings", []).append(msg)
        print("[warn] %s: %s" % (self.name, msg), flush=True)

    def result(self):
        failures = [k for k, v in self.checks.items() if not v["pass"]]
        return collections.OrderedDict([
            ("part", self.name),
            ("pass", bool(self.checks) and not failures),
            ("failures", failures),
            ("checks", self.checks),
            ("info", self.info),
            ("seconds", round(time.time() - self.t0, 2))])


def rel_diff(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(b), 1e-30)


def finite(*arrays):
    return all(bool(np.all(np.isfinite(np.asarray(a)))) for a in arrays)


def np_augment(x, flip, k):
    """Independent numpy reference of grouped_dataset.augment: flip_left_right,
    THEN rot90 (modules/dataset.py:59-81). Spatial axes are (-3, -2)."""
    if flip:
        x = x[..., :, ::-1, :]
    return np.ascontiguousarray(np.rot90(x, k, axes=(-3, -2)))


def op_types(concrete):
    """Every op type in a concrete function, nested library functions included
    (recompute_grad and tape gradients live there)."""
    gdef = concrete.graph.as_graph_def()
    types = set(n.op for n in gdef.node)
    for f in gdef.library.function:
        types.update(n.op for n in f.node_def)
    return types


def gpu_peak_mb():
    try:
        return tf.config.experimental.get_memory_info("GPU:0")["peak"] / float(1 << 20)
    except Exception:  # noqa: BLE001 - informational only
        return None


def gpu_current_mb():
    try:
        return tf.config.experimental.get_memory_info("GPU:0")["current"] / float(1 << 20)
    except Exception:  # noqa: BLE001
        return None


def gpu_reset_peak():
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
        return True
    except Exception:  # noqa: BLE001 - absent in some TF builds; the number is informational
        return False


# --------------------------------------------------------------------- PNG I/O

def png_bit_depth(path):
    with open(path, "rb") as f:
        head = f.read(33)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError("%s is not a PNG" % path)
    return int(head[24])


def read_png01(path):
    """PNG -> (H, W, 3) float32 in [0,1] via the training decoder, and its bit depth.

    tf.io.read_file + decode_png(channels=3), as modules/dataset.py:12-13 and
    grouped_dataset's parse. The divisor follows the file's bit depth exactly as
    MatUp's reader does (file_utils.py:58-61: uint16 / 65535, uint8 / 255); the
    eval GT height is 16-bit, every other map 8-bit. Grey PNGs come back as
    three equal channels (decode_png replicates), which decode_maps collapses
    by channel mean; MatUp reads channel 0 (file_utils.py:83-85) - the same value.
    """
    depth = png_bit_depth(path)
    raw = tf.io.read_file(path)
    if depth == 16:
        x = tf.cast(tf.image.decode_png(raw, channels=3, dtype=tf.uint16), tf.float32) / 65535.0
    elif depth == 8:
        x = tf.cast(tf.image.decode_png(raw, channels=3, dtype=tf.uint8), tf.float32) / 255.0
    else:
        raise ValueError("%s: bit depth %d" % (path, depth))
    return x.numpy(), depth


def committed_render_gt(eval_root, mat):
    """Newest MatUp run folder holding render_gt.png (sorted(...)[-1], the rule
    matup_render.py:165-172 and deeptex_compare.newest_run use)."""
    runs = sorted(d for d in glob.glob(os.path.join(eval_root, mat, "matup-output-%s" % mat, "*"))
                  if os.path.isfile(os.path.join(d, "render_gt.png")))
    return os.path.join(runs[-1], "render_gt.png") if runs else None


class Pairs(object):
    """The MatSynth training PNG pairs (<root>/<map>/{HR,LRx4}/<safe>__<map>.png)."""

    def __init__(self, root, scale, gt_size):
        self.root = root
        self.scale = scale
        self.gt_size = gt_size
        self._cache = {}
        self._pos_full = {}

    def metas(self):
        out = []
        for p in sorted(glob.glob(os.path.join(self.root, "_meta", "*.json"))):
            with open(p) as f:
                rec = json.load(f)
            out.append((rec.get("safe") or os.path.basename(p)[:-5], rec))
        if not out:
            raise RuntimeError("no _meta/*.json under %s" % self.root)
        return out

    def load(self, name, cache=True):
        """-> (lr5 uint8 (5, h, w, 3), hr5 uint8 (5, H, W, 3)), decoded as training does."""
        if name in self._cache:
            return self._cache[name]

        def rd(key, side):
            path = os.path.join(self.root, key, side, "%s__%s.png" % (name, key))
            if not os.path.isfile(path):
                raise RuntimeError("missing %s" % path)
            return tf.image.decode_png(tf.io.read_file(path), channels=3).numpy()

        lr5 = np.stack([rd(k, "LRx4") for k in MAP_KEYS])
        hr5 = np.stack([rd(k, "HR") for k in MAP_KEYS])
        if hr5.shape[1] != lr5.shape[1] * self.scale:
            raise RuntimeError("%s: HR %s is not LR %s x %d" % (name, hr5.shape, lr5.shape, self.scale))
        if cache:
            self._cache[name] = (lr5, hr5)
        return lr5, hr5

    def pos_full(self, side):
        if side not in self._pos_full:
            self._pos_full[side] = grouped_dataset.position_grid_np(side)
        return self._pos_full[side]

    def crop(self, name, off_lr, flip=False, rot_k=0, cache=True):
        """The training crop: grouped_dataset.crop_and_augment with a FIXED
        draw. Returns numpy (lr (5,s,s,3), hr (5,g,g,3), pos (g,g,2)) float32."""
        lr5, hr5 = self.load(name, cache)
        lr, hr, pos = grouped_dataset.crop_and_augment(
            lr5, hr5, self.pos_full(hr5.shape[1]), np.asarray(off_lr, np.int32),
            bool(flip), int(rot_k), self.scale, self.gt_size)
        return lr.numpy(), hr.numpy(), pos.numpy()


# ------------------------------------------------------------------ the context

class Ctx(object):
    """Everything the parts share: the render losses, their tf.functions, the
    training materials and the synthetic texels (each built once)."""

    def __init__(self, args, cfg, rcfg, gpu_name):
        self.args = args
        self.rcfg = rcfg
        self.gpu_name = gpu_name
        self.gt_size = int(cfg.get("gt_size", 128))
        self.scale = int(cfg.get("scale", 4))
        if self.gt_size % self.scale:
            die(2, "gt_size %d is not a multiple of scale %d" % (self.gt_size, self.scale))
        oy, ox = args.hr_offset
        if oy % self.scale or ox % self.scale:
            die(2, "--hr-offset %d,%d must be a multiple of scale %d" % (oy, ox, self.scale))
        self.off_lr = (oy // self.scale, ox // self.scale)
        self.pairs = Pairs(args.pairs_root, self.scale, self.gt_size)
        self._losses = {}
        self._fns = {}
        self._flow = None
        self._flow_error = None
        self._synthetic = None
        self.rl = self.render_loss(rcfg["light_chunks"])

    # -- render losses and their graphs
    def render_loss(self, chunks):
        chunks = int(chunks)
        if chunks not in self._losses:
            c = dict(self.rcfg)
            c["light_chunks"] = chunks
            self._losses[chunks] = render_tf.RenderLoss(c)
        return self._losses[chunks]

    def fn(self, rl, kind):
        """tf.function wrappers around one RenderLoss - the trainer runs the same
        Python code inside its own tf.function."""
        key = (id(rl), kind)
        if key in self._fns:
            return self._fns[key]

        def target(x5, pos):
            return rl.target(x5, pos)

        def loss(x5, pos, gt, valid):
            return rl.loss(x5, pos, gt, valid)

        def probe(sr5, hr5, pos):
            return rl.probe(sr5, hr5, pos)

        def display(x5, pos):
            return render_tf.render_display(rl.decode(x5, pos), rl.lights, rl.intensity)

        def unmasked(x5, pos, gt):
            return render_tf.l21(render_tf.render_display(rl.decode(x5, pos), rl.lights,
                                                          rl.intensity), gt)

        def loss_grad(x5, pos, gt, valid):
            # persistent, as deeptex_train's GAN step: recompute_grad is exercised
            # under the tape kind it will actually meet.
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x5)
                value = rl.loss(x5, pos, gt, valid)
            return value, tape.gradient(value, x5)

        def export(x5, pos):
            maps = rl.decode(x5, pos)
            radiance = render_tf.render_radiance(maps, rl.lights, rl.intensity)
            return maps, radiance, render_tf.process_raw_render(radiance)

        def eval_render(x5, pos, light):
            maps = rl.decode(x5, pos)
            return (render_tf.render_display(maps, light, rl.intensity),
                    maps["valid"], maps["nz_guarded"])

        table = {"target": target, "loss": loss, "probe": probe, "display": display,
                 "unmasked": unmasked, "loss_grad": loss_grad, "export": export,
                 "eval_render": eval_render}
        self._fns[key] = tf.function(table[kind])
        return self._fns[key]

    def rng(self, k):
        return np.random.RandomState([int(self.args.seed), int(k)])

    # -- data
    def valid_fraction(self, hr, pos):
        maps = render_tf.decode_maps(hr[None], pos[None], self.rcfg)
        return float(np.mean(maps["valid"].numpy()))

    def flow_batch(self):
        """Two training materials at the fixed crop, as one (B = 2) trainer batch:
        [textured metallic, plain dielectric], both with zero invalid GT texels.

        Rule (deterministic, sorted safe names): the first material whose _meta
        metallic is not constant and whose crop is 25-75% metal (> 0.5), and the
        first whose metallic is constant 0 with textured normal and roughness.
        --flow-materials overrides it.
        """
        if self._flow is not None:
            return self._flow
        if self._flow_error is not None:
            raise RuntimeError(self._flow_error)                     # do not rescan per part
        try:
            self._flow = self._pick_flow()
        except Exception as e:  # noqa: BLE001
            self._flow_error = "flow batch unavailable: %s: %s" % (type(e).__name__, e)
            raise
        return self._flow

    def _pick_flow(self):
        a = self.args
        rule = "--flow-materials"
        if a.flow_materials:
            names = [n for n in a.flow_materials.split(",") if n]       # count checked in main()
            metal_frac = None
        else:
            rule = ("first sorted: metallic not constant with %.0f-%.0f%% metal in the crop; "
                    "metallic constant 0 with textured normal/roughness; both all-valid"
                    % (100 * METAL_FRAC[0], 100 * METAL_FRAC[1]))
            metal = plain = None
            metal_frac = None
            scanned = 0
            for name, rec in self.pairs.metas():
                maps = rec.get("maps", {})
                mm = maps.get("metallic", {})
                want_metal = metal is None and mm.get("constant") is False
                want_plain = (plain is None and mm.get("constant") is True
                              and mm.get("hr_max") == 0
                              and maps.get("normal", {}).get("constant") is False
                              and maps.get("roughness", {}).get("constant") is False)
                if not (want_metal or want_plain):
                    continue
                scanned += 1
                if scanned > FLOW_SCAN_MAX:
                    break
                _, hr, pos = self.pairs.crop(name, self.off_lr, cache=False)
                if self.valid_fraction(hr, pos) < 1.0:
                    continue
                if want_metal:
                    frac = float(np.mean(hr[MTL].mean(axis=-1) > 0.5))
                    if METAL_FRAC[0] <= frac <= METAL_FRAC[1]:
                        metal, metal_frac = name, frac
                elif want_plain:
                    plain = name
                if metal and plain:
                    break
            if not (metal and plain):
                raise RuntimeError("flow material scan found metal=%s plain=%s after %d "
                                   "candidates; pass --flow-materials" % (metal, plain, scanned))
            names = [metal, plain]
        crops = [self.pairs.crop(n, self.off_lr) for n in names]
        lr = np.stack([c[0] for c in crops]).astype(np.float32)
        hr = np.stack([c[1] for c in crops]).astype(np.float32)
        pos = np.stack([c[2] for c in crops]).astype(np.float32)
        print("[*] flow batch: %s (%s)" % (names, rule), flush=True)
        return {"names": names, "rule": rule, "metal_frac": metal_frac,
                "lr": lr, "hr": hr, "pos": pos}

    def synthetic(self):
        """600 edge texels as one (1, 5, 1, 600, 3) stack: 20 normals x 6
        roughness x 5 metallic, albedo channels and height drawn (seeded) from
        their edge values with every value forced to appear."""
        if self._synthetic is not None:
            return self._synthetic
        normals = []
        normals.append(("nz=1", (0.0, 0.0, 1.0)))
        for nz in (0.5, 0.0, -0.3):
            r = np.sqrt(1.0 - nz * nz)
            for az in AZIMUTHS:
                t = np.radians(az)
                normals.append(("nz=%g/az=%d" % (nz, az), (r * np.cos(t), r * np.sin(t), nz)))
        normals.append(("zero", (0.0, 0.0, 0.0)))           # encoded (0.5, 0.5, 0.5)
        normals.append(("short", (0.03, 0.0, 0.04)))        # |n| = 0.05: masked, above the 0.01 floor
        normals.append(("long", (1.0, 1.0, 1.0)))           # |n| = sqrt 3, encoded (1, 1, 1)
        normals.append(("outside01", (1.4, 0.0, 0.8)))      # encoded (1.2, 0.5, 0.9)

        kinds, n_enc, r_enc, m_enc = [], [], [], []
        for kind, n in normals:
            enc = 0.5 * np.asarray(n, np.float64) + 0.5     # n_z = 0 -> exactly 0.5
            for r in ROUGH_ENC:
                for m in METAL_ENC:
                    kinds.append(kind)
                    n_enc.append(enc)
                    r_enc.append(r)
                    m_enc.append(m)
        n_tex = len(kinds)
        rng = self.rng(0)
        a_idx = rng.randint(0, len(ALBEDO_ENC), size=(n_tex, 3))
        h_idx = rng.randint(0, len(HEIGHT_ENC), size=n_tex)
        for c in range(3):
            a_idx[:len(ALBEDO_ENC), c] = np.arange(len(ALBEDO_ENC))
        h_idx[:len(HEIGHT_ENC)] = np.arange(len(HEIGHT_ENC))
        albedo = np.asarray(ALBEDO_ENC, np.float32)[a_idx]
        height = np.asarray(HEIGHT_ENC, np.float32)[h_idx]
        r_enc = np.asarray(r_enc, np.float32)
        m_enc = np.asarray(m_enc, np.float32)

        x5 = np.zeros((1, 5, 1, n_tex, 3), np.float32)
        x5[0, BC, 0] = albedo
        x5[0, NRM, 0] = np.asarray(n_enc, np.float32)
        x5[0, RGH, 0] = r_enc[:, None]
        x5[0, MTL, 0] = m_enc[:, None]
        x5[0, HGT, 0] = height[:, None]
        pos = rng.uniform(-1.0, 1.0, size=(1, 1, n_tex, 2)).astype(np.float32)
        self._synthetic = {
            "x5": x5, "pos": pos,
            "normal_kind": np.array(kinds), "zero_vector": np.array([k == "zero" for k in kinds]),
            "albedo_enc": albedo, "rough_enc": r_enc, "metal_enc": m_enc, "height_enc": height}
        return self._synthetic


def reference_like(x5):
    """A tame material of the same shape: the fixed target for the grads part."""
    ref = np.empty_like(x5)
    ref[:, BC] = 0.5
    ref[:, NRM] = np.array([0.5, 0.5, 1.0], np.float32)
    ref[:, RGH] = 0.5
    ref[:, MTL] = 0.0
    ref[:, HGT] = 0.5
    return ref


# ------------------------------------------------------------------- the parts

def part_export(ctx):
    p = Part("export")
    a = ctx.args
    rl = ctx.rl
    export_fn = ctx.fn(rl, "export")
    target_fn = ctx.fn(rl, "target")
    eval_fn = ctx.fn(rl, "eval_render")
    npz_path = os.path.join(a.out, "parity_tf.npz")
    if os.path.exists(npz_path):
        os.remove(npz_path)

    arrays = collections.OrderedDict()
    arrays["lights"] = rl.lights_np.astype(np.float32)
    arrays["eval_light"] = matup_lights.EVAL_LIGHT[None].astype(np.float32)
    arrays["map_keys"] = np.array(MAP_KEYS)
    # The torch side reads the decode constants (guard, mask, floor) from here.
    arrays["render_cfg"] = np.array(json.dumps(_clean(rl.cfg)))
    manifest = collections.OrderedDict([
        ("version", 1),
        ("created", datetime.datetime.utcnow().isoformat() + "Z"),
        ("script", "scripts/deeptex/render_gate.py --part export"),
        ("tf_version", tf.__version__), ("gpu", ctx.gpu_name),
        ("render_cfg", rl.cfg), ("n_lights", rl.n_lights),
        ("lights_sha256", hashlib.sha256(arrays["lights"].tobytes()).hexdigest()),
        ("flat_index", "n = (b*H + h)*W + w"),
        ("layouts", {"radiance": "(N,3,L) linear", "display": "(N,3,L) process_raw_render",
                     "encoded": "(B,5,H,W,3)", "encoded_flat": "(5,N,3)"}),
        ("graph", "one tf.function: rl.decode -> render_radiance -> process_raw_render "
                  "(the trainer's code, in graph mode, on %s)" % ctx.gpu_name),
        ("cases", collections.OrderedDict())])

    def run_case(case, x5, pos):
        maps, rad, disp = export_fn(tf.constant(x5), tf.constant(pos))
        maps = {k: v.numpy() for k, v in maps.items()}
        rad, disp = rad.numpy(), disp.numpy()
        t_disp, t_valid = target_fn(tf.constant(x5), tf.constant(pos))
        t_diff = float(np.max(np.abs(t_disp.numpy() - disp)))
        b, h, w = x5.shape[0], x5.shape[2], x5.shape[3]
        arrays[case + "__encoded"] = x5.astype(np.float32)
        arrays[case + "__encoded_flat"] = np.ascontiguousarray(
            np.transpose(x5, (1, 0, 2, 3, 4)).reshape(5, b * h * w, 3)).astype(np.float32)
        arrays[case + "__pos_xy"] = pos.astype(np.float32)
        for k in ("albedo", "metallic", "normal", "alpha", "pos", "valid", "nz_guarded"):
            arrays["%s__%s" % (case, k)] = maps[k]
        arrays[case + "__radiance"] = rad
        arrays[case + "__display"] = disp
        decoded_ok = finite(*[maps[k] for k in ("albedo", "metallic", "normal", "alpha", "pos")])
        nlen = np.linalg.norm(maps["normal"].astype(np.float64), axis=1)
        stats = collections.OrderedDict([
            ("B", b), ("H", h), ("W", w), ("N", b * h * w), ("L", rad.shape[2]),
            ("decoded_finite", decoded_ok),
            ("normal_len_max_dev", float(np.max(np.abs(nlen - 1.0)))),
            ("radiance_finite", finite(rad)), ("display_finite", finite(disp)),
            ("radiance_min", float(np.min(rad))), ("radiance_max", float(np.max(rad))),
            ("display_min", float(np.min(disp))), ("display_max", float(np.max(disp))),
            ("valid_frac", float(np.mean(maps["valid"]))),
            ("guarded", int(np.sum(maps["nz_guarded"]))),
            ("train_path_max_abs", t_diff),
            ("train_path_valid_equal", bool(np.array_equal(t_valid.numpy(), maps["valid"])))])
        manifest["cases"][case] = stats
        return stats

    # (ii) synthetic edge texels
    syn = ctx.synthetic()
    s = run_case("synthetic", syn["x5"], syn["pos"])
    for k in ("normal_kind", "zero_vector", "albedo_enc", "rough_enc", "metal_enc", "height_enc"):
        arrays["synthetic__" + k] = syn[k]
    manifest["cases"]["synthetic"]["kind"] = "synthetic"
    p.check("synthetic_finite", s["decoded_finite"] and s["radiance_finite"] and s["display_finite"],
            **{k: s[k] for k in ("N", "radiance_max", "display_max", "guarded", "valid_frac")})
    p.check("synthetic_ranges",
            s["radiance_min"] >= 0.0 and 0.0 <= s["display_min"]
            and s["display_max"] <= 1.0 + DISPLAY_MAX_SLACK and s["normal_len_max_dev"] <= 1e-5,
            radiance_min=s["radiance_min"], display_min=s["display_min"],
            display_max=s["display_max"], normal_len_max_dev=s["normal_len_max_dev"])
    p.info["synthetic_train_path_max_abs"] = s["train_path_max_abs"]

    # (i) anchored crops of the eval GT, plus the full-material eval-light render
    oy, ox = a.hr_offset
    g = ctx.gt_size
    anchored = [m for m in a.anchored.split(",") if m]
    arrays["anchored_materials"] = np.array(anchored)
    for mat in anchored:
        case = "anchored_" + mat
        gt_dir = os.path.join(a.eval_root, mat, "GT")
        planes, depths = [], []
        for k in MAP_KEYS:
            x, d = read_png01(os.path.join(gt_dir, k + ".png"))
            planes.append(x)
            depths.append(d)
        x5_full = np.stack(planes)[None].astype(np.float32)            # (1, 5, S, S, 3)
        side = x5_full.shape[2]
        if x5_full.shape[3] != side or oy + g > side or ox + g > side:
            raise RuntimeError("%s: GT is %s, cannot crop %d at (%d, %d)"
                               % (mat, x5_full.shape, g, oy, ox))
        pos_full = grouped_dataset.position_grid_np(side)
        x5 = np.ascontiguousarray(x5_full[:, :, oy:oy + g, ox:ox + g])
        pos = np.ascontiguousarray(pos_full[oy:oy + g, ox:ox + g])[None]
        st = run_case(case, x5, pos)
        arrays[case + "__material"] = np.array(mat)
        arrays[case + "__offset"] = np.array([oy, ox], np.int32)                # HR (y, x)
        arrays[case + "__bit_depth"] = np.array(depths, np.int32)
        st["kind"] = "anchored"
        st["material"] = mat
        st["hr_offset"] = [oy, ox]
        st["bit_depth"] = dict(zip(MAP_KEYS, depths))

        ok = st["decoded_finite"] and st["radiance_finite"] and st["display_finite"]
        p.check("anchored_finite[%s]" % mat, ok and st["display_max"] <= 1.0 + DISPLAY_MAX_SLACK
                and st["radiance_min"] >= 0.0,
                display_min=st["display_min"], display_max=st["display_max"],
                valid_frac=st["valid_frac"], guarded=st["guarded"])
        # The npz must hold what the TRAINER computes: the export graph (which
        # also fetches maps and radiance) against RenderLoss.target's graph.
        p.check("anchored_train_path[%s]" % mat,
                st["train_path_max_abs"] <= EXPORT_TRAIN_PATH_MAX and st["train_path_valid_equal"],
                max_abs=st["train_path_max_abs"], tol=EXPORT_TRAIN_PATH_MAX)
        if st["guarded"]:
            p.warn("%s: the n_z guard moved %d GT texels in the crop; the torch side "
                   "must apply the same guard and mask" % (mat, st["guarded"]))

        # Full material under EVAL_LIGHT, cropped like render_gt.png.
        e_disp, e_valid, e_guard = eval_fn(tf.constant(x5_full), tf.constant(pos_full[None]),
                                           tf.constant(arrays["eval_light"]))
        b0 = EVAL_BORDER
        e_disp = e_disp.numpy().reshape(side, side, 3)[b0:side - b0, b0:side - b0]
        e_valid = e_valid.numpy().reshape(side, side)[b0:side - b0, b0:side - b0]
        e_guard = e_guard.numpy().reshape(side, side)[b0:side - b0, b0:side - b0]
        arrays[case + "__eval_display"] = np.ascontiguousarray(e_disp)
        arrays[case + "__eval_valid"] = np.ascontiguousarray(e_valid)
        arrays[case + "__eval_nz_guarded"] = np.ascontiguousarray(e_guard)
        arrays[case + "__eval_border"] = np.array(b0, np.int32)
        p.check("eval_render_finite[%s]" % mat, finite(e_disp),
                shape=list(e_disp.shape), guarded=int(e_guard.sum()),
                valid_frac=float(e_valid.mean()))

        # Informational: against the committed render_gt.png (16-bit, written by
        # truncation, file_utils.py:112-125). render_parity_torch.py owns the verdict.
        ref_path = committed_render_gt(a.eval_root, mat)
        cmp_ = collections.OrderedDict([("file", ref_path)])
        if ref_path:
            cmp_["bit_depth"] = png_bit_depth(ref_path)
            ref = tf.image.decode_png(tf.io.read_file(ref_path), channels=3,
                                      dtype=tf.uint16).numpy()
            cmp_["shape"] = list(ref.shape)
            if cmp_["bit_depth"] != 16:
                p.warn("%s: %s is %d-bit, expected 16 (file_utils.py:115)"
                       % (mat, ref_path, cmp_["bit_depth"]))
            elif ref.shape == e_disp.shape:
                m = e_valid > 0.5
                lsb = (np.floor(65535.0 * np.clip(e_disp.astype(np.float64), 0.0, 1.0))
                       - ref.astype(np.float64))
                dfl = np.abs(e_disp.astype(np.float64) - ref.astype(np.float64) / 65535.0)
                cmp_["valid_texels"] = int(m.sum())
                cmp_["max_abs_valid"] = float(dfl[m].max()) if m.any() else None
                cmp_["max_lsb_valid"] = float(np.abs(lsb[m]).max()) if m.any() else None
                cmp_["frac_lsb_over_2_valid"] = float(np.mean(np.abs(lsb[m]) > 2)) if m.any() else None
                cmp_["max_lsb_all"] = float(np.abs(lsb).max())
                if cmp_["max_lsb_valid"] is not None:
                    print("[export] %s vs committed render_gt.png: max %.3g (%.0f LSB) on %d "
                          "valid texels" % (mat, cmp_["max_abs_valid"], cmp_["max_lsb_valid"],
                                            cmp_["valid_texels"]), flush=True)
                    if cmp_["max_lsb_valid"] > 2:
                        p.warn("%s: eval-light render differs from the committed render_gt.png "
                               "by %d LSB (> 2) on valid texels" % (mat, cmp_["max_lsb_valid"]))
            else:
                p.warn("%s: render_gt.png is %s, the cropped TF render %s"
                       % (mat, ref.shape, e_disp.shape))
        else:
            p.warn("%s: no committed render_gt.png under %s" % (mat, a.eval_root))
        st["committed_render_gt"] = cmp_

    arrays["cases"] = np.array(list(manifest["cases"].keys()))
    arrays["manifest"] = np.array(json.dumps(_clean(manifest), allow_nan=False))
    tmp = npz_path + ".partial"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, npz_path)
    digest = sha256(npz_path)
    p.info["npz"] = npz_path
    p.info["npz_sha256"] = digest
    p.info["npz_bytes"] = os.path.getsize(npz_path)
    p.info["cases"] = manifest["cases"]
    p.check("npz_written", os.path.getsize(npz_path) > 0, bytes=p.info["npz_bytes"],
            sha256=digest[:16], cases=len(manifest["cases"]))
    return p.result()


def part_grads(ctx):
    p = Part("grads")
    syn = ctx.synthetic()
    x5, pos = syn["x5"], syn["pos"]
    ref = reference_like(x5)
    kinds = syn["normal_kind"]
    for chunks in (1, 4):
        rl = ctx.render_loss(chunks)
        gt, valid = ctx.fn(rl, "target")(tf.constant(ref), tf.constant(pos))
        var = tf.Variable(x5, name="edge_texels")
        value, grad = ctx.fn(rl, "loss_grad")(var, tf.constant(pos), gt, valid)
        g = grad.numpy().astype(np.float64)
        ok_finite = finite(value.numpy(), g)
        norm = float(np.sqrt(np.sum(np.where(np.isfinite(g), g, 0.0) ** 2)))
        per_texel = np.abs(g[0]).max(axis=(0, 1, 3))                    # (N,) over slots, channels
        per_kind = collections.OrderedDict()
        for k in collections.OrderedDict.fromkeys(kinds.tolist()):
            sel = kinds == k
            per_kind[k] = float(np.nanmax(per_texel[sel]))
        worst = int(np.nanargmax(per_texel)) if np.isfinite(per_texel).any() else -1
        p.check("synthetic_chunks%d" % chunks, ok_finite and norm < GRAD_NORM_MAX,
                loss=float(value.numpy()), grad_norm=norm, tol=GRAD_NORM_MAX,
                grad_max_abs=float(np.nanmax(np.abs(g))),
                nonfinite=int(np.sum(~np.isfinite(g))),
                worst_kind=str(kinds[worst]) if worst >= 0 else None)
        p.info["synthetic_chunks%d_grad_max_by_normal_kind" % chunks] = per_kind

    # UNCLIPPED: how much of a plausible unbounded SR output gets no render gradient.
    fb = ctx.flow_batch()
    rl = ctx.rl
    hr, pos_b = tf.constant(fb["hr"]), tf.constant(fb["pos"])
    gt, valid = ctx.fn(rl, "target")(hr, pos_b)
    x = fb["hr"] + ctx.rng(1).normal(0.0, NOISE_UNCLIPPED, fb["hr"].shape).astype(np.float32)
    value, grad = ctx.fn(rl, "loss_grad")(tf.constant(x), pos_b, gt, valid)
    g = grad.numpy()
    norm = float(np.sqrt(np.sum(g.astype(np.float64) ** 2)))
    p.check("unclipped_finite", finite(value.numpy(), g) and norm < GRAD_NORM_MAX,
            loss=float(value.numpy()), grad_norm=norm, materials=fb["names"])
    zero = collections.OrderedDict()
    for s, key in enumerate(MAP_KEYS):
        gz = g[:, s] == 0.0
        oor = (x[:, s] < 0.0) | (x[:, s] > 1.0)
        zero[key] = {"elements_zero": float(gz.mean()), "texels_all_zero": float(gz.all(-1).mean()),
                     "elements_outside01": float(oor.mean())}
    p.info["unclipped_zero_grad"] = zero
    p.info["unclipped_valid_frac"] = float(np.mean(valid.numpy()))
    print("[grads] unclipped zero-gradient texels: %s" % ", ".join(
        "%s=%.3f" % (k, v["texels_all_zero"]) for k, v in zero.items()), flush=True)
    return p.result()


def part_flow(ctx):
    p = Part("flow")
    fb = ctx.flow_batch()
    rl = ctx.rl
    hr_np, pos_np = fb["hr"], fb["pos"]
    hr, pos = tf.constant(hr_np), tf.constant(pos_np)
    p.info["materials"] = fb["names"]
    p.info["rule"] = fb["rule"]
    p.info["metal_frac_first"] = fb["metal_frac"]
    p.info["hr_offset"] = list(ctx.args.hr_offset)

    # l2,1(GT, GT): the monitor path (target + loss in ONE graph, as the trainer).
    floor = float(np.sqrt(np.float32(render_tf.L21_EPS)))
    l_gg = float(ctx.fn(rl, "probe")(hr, hr, pos).numpy())
    gt, valid = ctx.fn(rl, "target")(hr, pos)
    disp = ctx.fn(rl, "display")(hr, pos).numpy()
    d = disp - gt.numpy()
    above = int(np.sum(np.sum(d.astype(np.float64) ** 2, axis=1) > render_tf.L21_EPS))
    p.check("floor_gt_vs_gt", rel_diff(l_gg, floor) <= FLOOR_REL, value=l_gg, expected=floor,
            rel=rel_diff(l_gg, floor), tol=FLOOR_REL,
            two_graph_max_abs=float(np.max(np.abs(d))), two_graph_pairs_above_floor=above)
    p.info["valid_frac"] = float(np.mean(valid.numpy()))

    # Gradient reaches every map (but height).
    sr = np.clip(hr_np + ctx.rng(2).normal(0.0, NOISE_FLOW, hr_np.shape).astype(np.float32),
                 0.0, 1.0).astype(np.float32)
    value, grad = ctx.fn(rl, "loss_grad")(tf.constant(sr), pos, gt, valid)
    g = grad.numpy().astype(np.float64)
    p.check("grad_finite", finite(value.numpy(), g), loss=float(value.numpy()))
    per_mat = np.abs(g).mean(axis=(2, 3, 4))                             # (B, 5)
    rows = [(n, per_mat[i]) for i, n in enumerate(fb["names"])]
    rows.append(("batch", np.abs(g).mean(axis=(0, 2, 3, 4))))
    for name, v in rows:
        nums = dict(("mean_abs_" + k, float(v[s])) for s, k in enumerate(MAP_KEYS))
        p.check("grad_reaches_maps[%s]" % name, all(v[s] > 0 for s in (BC, NRM, RGH, MTL)), **nums)
        ref = float(np.median([v[BC], v[NRM], v[RGH]]))
        ratio = float(v[HGT]) / ref if ref > 0 else float("inf")
        ref4 = float(np.median([v[BC], v[NRM], v[RGH], v[MTL]]))
        p.check("height_small[%s]" % name, ratio <= HEIGHT_RATIO_MAX, ratio=ratio,
                tol=HEIGHT_RATIO_MAX, ref_median_bc_nrm_rgh=ref,
                ratio_vs_median4=float(v[HGT]) / ref4 if ref4 > 0 else float("inf"))

    # Descent: a small step along -g/||g|| lowers the loss (both values from one graph).
    gn = float(np.sqrt(np.sum(g ** 2)))
    loss_fn = ctx.fn(rl, "loss")
    x2 = (sr.astype(np.float64) - DESCENT_STEP * g / max(gn, 1e-30)).astype(np.float32)
    l0 = float(loss_fn(tf.constant(sr), pos, gt, valid).numpy())
    l1 = float(loss_fn(tf.constant(x2), pos, gt, valid).numpy())
    p.check("descent", l1 < l0, loss_before=l0, loss_after=l1, step=DESCENT_STEP,
            grad_norm=gn, first_order_drop=DESCENT_STEP * gn)
    return p.result()


def part_grid(ctx):
    p = Part("grid")
    fb = ctx.flow_batch()
    name = fb["names"][0]
    rl = ctx.rl
    display = ctx.fn(rl, "display")
    g = ctx.gt_size
    n_l = rl.n_lights
    _, hr5 = ctx.pairs.load(name)
    pos_full = ctx.pairs.pos_full(hr5.shape[1])
    p.info["material"] = name

    def render_img(hr, pos):
        out = display(tf.constant(hr[None]), tf.constant(pos[None]))      # (g*g, 3, L)
        return tf.reshape(out, [g, g, 3 * n_l])

    for oy, ox in GRID_OFFSETS_LR:
        lr_id, hr_id, pos_id = ctx.pairs.crop(name, (oy, ox))
        s = ctx.scale
        want_pos = pos_full[s * oy:s * oy + g, s * ox:s * ox + g]
        p.check("identity_pos[%d,%d]" % (oy, ox), np.array_equal(pos_id, want_pos),
                max_abs=float(np.max(np.abs(pos_id - want_pos))))
        ref = render_img(hr_id, pos_id)
        for flip in (False, True):
            for k in range(4):
                lr_a, hr_a, pos_a = ctx.pairs.crop(name, (oy, ox), flip, k)
                exact = (np.array_equal(lr_a, np_augment(lr_id, flip, k))
                         and np.array_equal(hr_a, np_augment(hr_id, flip, k))
                         and np.array_equal(pos_a, np_augment(pos_id, flip, k)))
                got = render_img(hr_a, pos_a).numpy()
                want = grouped_dataset.augment(ref, flip, k).numpy()
                err = float(np.max(np.abs(got - want)))
                p.check("equivariance[%d,%d flip=%d k=%d]" % (oy, ox, int(flip), k),
                        exact and err <= GRID_TOL, max_abs=err, tol=GRID_TOL, crops_exact=exact)
    return p.result()


def part_chunks(ctx):
    p = Part("chunks")
    fb = ctx.flow_batch()
    rl1, rl4 = ctx.render_loss(1), ctx.render_loss(4)
    hr, pos = tf.constant(fb["hr"]), tf.constant(fb["pos"])
    gt, valid = ctx.fn(rl1, "target")(hr, pos)
    sr = tf.constant(np.clip(fb["hr"] + ctx.rng(3).normal(0.0, NOISE_FLOW, fb["hr"].shape)
                             .astype(np.float32), 0.0, 1.0).astype(np.float32))
    out = {}
    for name, rl in (("chunks1", rl1), ("chunks4", rl4)):
        f = ctx.fn(rl, "loss_grad")
        f(sr, pos, gt, valid)                                          # trace + warm up
        base = gpu_current_mb()
        reset = gpu_reset_peak()
        value, grad = f(sr, pos, gt, valid)
        value, grad = float(value.numpy()), grad.numpy()
        peak = gpu_peak_mb() if reset else None
        out[name] = (value, grad)
        p.info["%s_peak_mb" % name] = peak
        p.info["%s_peak_above_baseline_mb" % name] = (peak - base) if (peak is not None
                                                                     and base is not None) else None
    (l1, g1), (l4, g4) = out["chunks1"], out["chunks4"]
    p.check("loss_equal", finite(l1, l4) and rel_diff(l4, l1) <= CHUNK_LOSS_REL,
            loss_chunks1=l1, loss_chunks4=l4, rel=rel_diff(l4, l1), tol=CHUNK_LOSS_REL)
    gmax = float(np.max(np.abs(g1)))
    grel = float(np.max(np.abs(g4.astype(np.float64) - g1))) / max(gmax, 1e-30)
    p.check("grad_equal", finite(g1, g4) and grel <= CHUNK_GRAD_REL,
            max_abs_diff_over_max_abs=grel, tol=CHUNK_GRAD_REL, grad_max_abs=gmax)
    print("[chunks] peak GPU memory for loss+grad: chunks1 %s MB, chunks4 %s MB (above baseline "
          "%s / %s)" % (p.info["chunks1_peak_mb"], p.info["chunks4_peak_mb"],
                        p.info["chunks1_peak_above_baseline_mb"],
                        p.info["chunks4_peak_above_baseline_mb"]), flush=True)
    return p.result()


def part_mask(ctx):
    p = Part("mask")
    rl = ctx.rl
    target_fn, loss_fn = ctx.fn(rl, "target"), ctx.fn(rl, "loss")
    unmasked_fn, display_fn = ctx.fn(rl, "unmasked"), ctx.fn(rl, "display")

    # Nothing invalid: the mask must be a no-op.
    fb = ctx.flow_batch()
    hr, pos = tf.constant(fb["hr"]), tf.constant(fb["pos"])
    gt, valid = target_fn(hr, pos)
    vf = float(np.mean(valid.numpy()))
    sr = tf.constant(np.clip(fb["hr"] + ctx.rng(4).normal(0.0, NOISE_FLOW, fb["hr"].shape)
                             .astype(np.float32), 0.0, 1.0).astype(np.float32))
    lm = float(loss_fn(sr, pos, gt, valid).numpy())
    lu = float(unmasked_fn(sr, pos, gt).numpy())
    p.check("clean_masked_equals_unmasked", vf == 1.0 and rel_diff(lm, lu) <= MASK_REL,
            materials=fb["names"], valid_frac=vf, masked=lm, unmasked=lu,
            rel=rel_diff(lm, lu), tol=MASK_REL)

    # Materials with invalid GT texels: report the fraction, verify the formula.
    for name in [m for m in ctx.args.mask_materials.split(",") if m]:
        _, hr1, pos1 = ctx.pairs.crop(name, ctx.off_lr)
        hr1, pos1 = tf.constant(hr1[None]), tf.constant(pos1[None])
        gt1, v1 = target_fn(hr1, pos1)
        v = v1.numpy().astype(np.float64)
        sr1 = tf.constant(np.clip(hr1.numpy() + ctx.rng(5).normal(0.0, NOISE_FLOW, hr1.shape)
                                  .astype(np.float32), 0.0, 1.0).astype(np.float32))
        lm1 = float(loss_fn(sr1, pos1, gt1, v1).numpy())
        lu1 = float(unmasked_fn(sr1, pos1, gt1).numpy())
        d = display_fn(sr1, pos1).numpy().astype(np.float64) - gt1.numpy().astype(np.float64)
        e = np.sqrt(np.maximum(np.sum(d * d, axis=1), render_tf.L21_EPS))          # (N, L)
        manual = float(np.sum(v * e) / (max(np.sum(v), 1.0) * e.shape[1]))
        inv = 1.0 - float(v.mean())
        if inv == 0.0:
            p.warn("%s has no invalid GT texel in the crop - the masked branch is untested "
                   "on it" % name)
        ok = finite(lm1) and (rel_diff(lm1, manual) <= MASK_FORMULA_REL
                              if manual > 0 else abs(lm1) <= 1e-12)
        p.check("invalid_formula[%s]" % name, ok, invalid_frac=inv, masked=lm1, unmasked=lu1,
                manual_f64=manual, rel=rel_diff(lm1, manual) if manual > 0 else None,
                tol=MASK_FORMULA_REL)
    return p.result()


def part_graph(ctx):
    p = Part("graph")
    p.check("map_slots_match", tuple(render_tf.MAP_SLOTS) == tuple(grouped_dataset.MAP_KEYS),
            render_tf=list(render_tf.MAP_SLOTS), grouped_dataset=list(grouped_dataset.MAP_KEYS))
    b, g = 2, ctx.gt_size
    f32 = tf.float32
    for chunks in (1, 4):
        rl = ctx.render_loss(chunks)
        n, n_l = b * g * g, rl.n_lights
        x5 = tf.TensorSpec([b, 5, g, g, 3], f32)
        pos = tf.TensorSpec([b, g, g, 2], f32)
        gt = tf.TensorSpec([n, 3, n_l], f32)
        valid = tf.TensorSpec([n, 1], f32)
        conc = collections.OrderedDict()
        if chunks == 1:
            conc["target"] = ctx.fn(rl, "target").get_concrete_function(x5, pos)
            conc["display"] = ctx.fn(rl, "display").get_concrete_function(x5, pos)
            side = grouped_dataset.HR_SIZE
            conc["eval_render"] = ctx.fn(rl, "eval_render").get_concrete_function(
                tf.TensorSpec([1, 5, side, side, 3], f32), tf.TensorSpec([1, side, side, 2], f32),
                tf.TensorSpec([1, 3], f32))
        conc["loss_chunks%d" % chunks] = ctx.fn(rl, "loss").get_concrete_function(x5, pos, gt, valid)
        conc["loss_grad_chunks%d" % chunks] = ctx.fn(rl, "loss_grad").get_concrete_function(
            x5, pos, gt, valid)
        for name, cf in conc.items():
            types = op_types(cf)
            bad = sorted(types & FORBIDDEN_OPS)
            p.check("no_matmul[%s]" % name, not bad, op_types=len(types), forbidden=bad)
    return p.result()


PART_FUNCS = {"export": part_export, "grads": part_grads, "flow": part_flow,
              "grid": part_grid, "chunks": part_chunks, "mask": part_mask, "graph": part_graph}


# ------------------------------------------------------------------------ main

def parse_offset(s):
    try:
        y, x = (int(v) for v in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("want 'y,x', got %r" % s)
    if y < 0 or x < 0:
        raise argparse.ArgumentTypeError("offset must be >= 0")
    return (y, x)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all",
                    help="comma-separated subset of %s, or 'all'" % ",".join(PARTS))
    ap.add_argument("--cfg", default=DEFAULT_CFG, help="only its render: block (and gt_size/scale) is read")
    ap.add_argument("--eval-root", default="/workspace/runs/matsynth")
    ap.add_argument("--pairs-root", default="/workspace/runs/deeptex/datasets/matsynth")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--anchored", default=",".join(ANCHORED),
                    help="eval materials for the anchored export crops")
    ap.add_argument("--hr-offset", type=parse_offset, default=HR_OFFSET,
                    help="HR crop offset 'y,x' in the 512 grid (multiple of scale)")
    ap.add_argument("--flow-materials", default="",
                    help="two safe names (textured metal first) instead of the _meta scan")
    ap.add_argument("--mask-materials", default=",".join(MASK_INVALID),
                    help="training materials with invalid GT normals for the mask part")
    ap.add_argument("--seed", type=int, default=0, help="numpy seed for noise and synthetic draws")
    args = ap.parse_args()

    parts = list(PARTS) if args.part == "all" else [s for s in args.part.split(",") if s]
    unknown = [s for s in parts if s not in PARTS]
    if unknown or not parts:
        die(2, "--part %r: unknown %s; choose from %s or all" % (args.part, unknown, ",".join(PARTS)))
    args.out = refuse_external(args.out, "--out")
    if args.flow_materials and len([n for n in args.flow_materials.split(",") if n]) != 2:
        die(2, "--flow-materials wants exactly two safe names (textured metal first)")
    # Never leave a stale PASS (or a stale npz) behind a run that dies part-way.
    os.makedirs(args.out, exist_ok=True)
    stale = [os.path.join(args.out, p + ".json") for p in parts]
    if "export" in parts:
        stale.append(os.path.join(args.out, "parity_tf.npz"))
    for path in stale:
        if os.path.exists(path):
            os.remove(path)

    # GPU before anything touches it: memory growth must be set first, and the
    # parity numbers are only meaningful for the GPU path.
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        die(6, "no GPU visible to TensorFlow. The parity numbers describe the GPU path "
               "the trainer runs (TF32, Grappler); a CPU run would certify something else.")
    for gdev in gpus:
        try:
            tf.config.experimental.set_memory_growth(gdev, True)
        except RuntimeError as e:
            print("[warn] set_memory_growth: %s" % e)
    try:
        gpu_name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", gpus[0].name)
    except Exception:  # noqa: BLE001
        gpu_name = gpus[0].name

    if not os.path.isfile(args.cfg):
        die(2, "--cfg %s not found" % args.cfg)
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg.get("render"), dict):
        die(2, "%s has no render: block" % args.cfg)
    try:
        rcfg = render_tf.validate_render_cfg(cfg["render"])
    except ValueError as e:
        die(2, "%s: %s" % (args.cfg, e))

    ctx = Ctx(args, cfg, rcfg, gpu_name)
    meta = collections.OrderedDict([
        ("created", datetime.datetime.utcnow().isoformat() + "Z"),
        ("script", "scripts/deeptex/render_gate.py"),
        ("tf_version", tf.__version__), ("gpu", gpu_name),
        ("cfg", os.path.abspath(args.cfg)), ("render_cfg", rcfg),
        ("gt_size", ctx.gt_size), ("scale", ctx.scale), ("seed", args.seed)])
    print("[*] render gate: parts=%s gpu=%s cfg=%s out=%s"
          % (",".join(parts), gpu_name, args.cfg, args.out), flush=True)

    results = collections.OrderedDict()
    for part in parts:
        path = os.path.join(args.out, part + ".json")
        print("\n[gate] ---- %s ----" % part, flush=True)
        try:
            res = PART_FUNCS[part](ctx)
        except Exception as e:  # noqa: BLE001 - record and go on to the next part
            traceback.print_exc()
            res = collections.OrderedDict([("part", part), ("pass", False),
                                           ("failures", ["exception"]),
                                           ("error", "%s: %s" % (type(e).__name__, e))])
        res.update(meta)
        write_json(path, res)
        results[part] = res
        print("[gate] part=%s pass=%s failures=%s -> %s"
              % (part, res["pass"], ",".join(res.get("failures", [])) or "-", path), flush=True)

    failed = [k for k, v in results.items() if not v["pass"]]
    print("\n[gate] %s" % ("PASS (%s)" % ",".join(parts) if not failed
                           else "FAIL: %s" % ",".join(failed)), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
