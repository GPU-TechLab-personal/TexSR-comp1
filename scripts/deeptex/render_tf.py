"""MatUp's SVBRDF renderer, ported line for line to TensorFlow, plus the l2,1 loss.

Runs in the deeptex container (TF 2.10.1, Python 3.8). Imported lazily by
scripts/deeptex/deeptex_train.py in grouped mode, and by render_gate.py and
render_calibrate.py.

WHY A PORT. MatUp's renderer exists only in torch
(external/matup_egsr/matup/renderer.py) and the deeptex image may not gain
packages, so the render-space loss has to be rebuilt in TF. It is a literal port,
not a re-derivation: every statement below cites the renderer.py /
render_utils.py / data_utils.py line it mirrors, keeps torch's operand order
(`a * x * x` stays `(a * x) * x`), and keeps every clamp where MatUp put it. Gate
(b) (render_gate.py --part export + render_parity_torch.py) holds it to MatUp's
own torch code with a STATISTICAL bound against a float64 oracle: outside
Heaviside ties, at most max(20, 0.1%) of display values may exceed 2x torch's
own float32 error + 1e-6, and none by more than 2e-5 (Grappler's rsqrt rewrite
rounds differently from torch, so a per-value 2x rule fails correct code).

THINGS THAT ARE EASY TO GET SUBTLY WRONG, and what this file does about them:

  * TF32. TF 2.10 runs MatMul/Einsum/BatchMatMul in TF32 on Ampere by default,
    and nothing in this repo turns that off (turning it off globally would also
    change G/D numerics against matsynth_all). Near a highlight at alpha ~ 2e-3,
    D is so sharp that TF32 dot products move it by orders of magnitude. So
    every dot product here is `reduce_sum(x * y)`, exactly as torch does it
    (render_utils.py:7-8). Never rewrite one as matmul/einsum/tensordot; gate
    (b) "graph" asserts none appears.
  * c_sqr. MatUp calls its anisotropic GGX with c_sqr = 0 * roughness
    (renderer.py:174). Every term c enters adds an exact +-0.0, so the terms are
    dropped with bit-identical results and ~10 fewer (N,1,L) intermediates.
  * Heavisides are taken on the RAW dots, before the [1e-8, 0.9999] clamp
    (renderer.py:129-135); D/V then use the clamped dots but the UNCLAMPED
    tangent-frame projections. Swapping either for the textbook form changes
    grazing-light values.
  * Nothing is created at import time. A tf.constant here would initialise the
    GPU before deeptex_train.py's set_memory_growth() - upstream's version of
    which swallows the RuntimeError (modules/utils.py:33-35) - and memory growth
    would be silently lost. Constants are Python floats; the light table is
    built in RenderLoss.__init__.

DECODE - one function for SR, GT and every gate, so a decode choice can never
differ between the two sides of the loss. MatUp's own path for a MatSynth folder
is matup_render.load_set (scripts/matup/matup_render.py:53-76) +
position3d (:79-83) + get_renorm_clamped_maps (data_utils.py:74-88). Slot order
is MAP_SLOTS, the same as grouped_dataset.MAP_KEYS.

  baseColor  clip [0,1], then sRGB -> linear (file_utils.py:91-98). For GT this
             equals MatUp (the PNG is already in [0,1]); for the unbounded SR
             output the clip happens before the curve, which gives the same
             forward value as MatUp's clamp after it (renderer.py:104).
  normal     n = 2x - 1 (dataLoader.py:73), normalise, y -> -y (OpenGL ->
             MatUp's internal frame, dataLoader.py:74-75), n_z guard, normalise
             again (data_utils.py:81; renderer.py:170 normalises once more).
             OURS, not MatUp's:
               - the first normalisation floors sum(n^2) at 1e-4 (|n| >= 0.01).
                 MatUp's GT has no near-zero vectors; an SR texel near encoded
                 (0.5,0.5,0.5) would otherwise amplify its gradient up to 1e6x.
               - n_z = max(n_z, 1e-3). branchlessONB (render_utils.py:96-97)
                 computes sign(n_z) and 1/(sign + n_z), which is 0/0 at n_z == 0
                 exactly. SR outputs can hit that; PNG-decoded GT cannot.
               - the encoded SR normal is NOT clipped to [0,1] first, so its
                 gradient survives. deeptex_compare.py clips before writing the
                 delivered 8-bit PNG, so train and eval directions differ where
                 SR encodes outside [0,1]. Declared, not fixed.
  roughness  channel mean, clip [0,1], floor 0.045 (dataLoader.py:39-40), then
             alpha = clip(r^2, 2e-3, 1) (dataLoader.py:114, matup_render.py:73-74).
             The renderer squares alpha AGAIN (renderer.py:172-173); that is
             MatUp, not a bug.
  metallic   channel mean, clip [0,1] (data_utils.py:79).
  height     channel mean, clip [0,1]; z = 0.01 * (h - 0.5) (dataLoader.py:115,
             154). The clip is ours (position3d never clips) and only ever bites
             on SR.
  The channel mean equals channel 0 (what MatUp reads) on every grey GT PNG, and
  equals deeptex_compare's --scalar-collapse mean on the eval side.

  valid      GT only: 1 where |2x - 1| >= 0.1 AND the unit n_z (before the guard)
             >= 1e-3. 0.18% of the MatSynth training GT texels are back-facing or
             garbage (e.g. the *_fur_cow_long maps are a constant RGB); the guard
             would turn them into horizontal normals that grazing lights
             brighten, i.e. invent the target. They are masked out of the loss
             instead. MatUp has no mask: its target is a set of renders, not
             GT maps, so the problem never arises there.

WHAT IS NOT PORTED: MatUp's metallic regulariser (train.py:23-27, alpha_reg 0.1)
pulls the optimised metallic towards the bilinear-upsampled LR metallic it
started from (the tile centre of its input); a generator has no such per-texel
starting point, so it is left out and must be declared. Light falloff is off in
every MatUp entry point this mirrors (main.py:56 default, matup_render.py:217),
so it is not implemented.
"""

import os
import sys

import numpy as np
import tensorflow as tf

_COMMON = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        os.pardir, "common"))
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

import matup_lights  # noqa: E402  (numpy only; see that file for why it is not imported from MatUp)


# ------------------------------------------------------------------- constants
# Python floats only - TF converts each to a float32 constant at trace time, the
# same float32 value torch uses when a Python scalar meets a float32 tensor.

INTENSITY = 8.0            # renderer.py:14   Renderer.default_light_intensity
MIN_ROUGHNESS = 0.045      # renderer.py:15   applied at load: dataLoader.py:39-40
MIN_ALPHA = 2e-3           # renderer.py:16   dataLoader.py:114, data_utils.py:83
VIS_MIN = 1e-8             # renderer.py:19   min_visibility
NDF_DENOM_MIN = 1e-8       # renderer.py:20   min_ndf_denom
ONE_OVER_PI = 1 / np.pi    # renderer.py:21   (a Python float: np.pi is one)
F0_DIELECTRIC = 0.04       # renderer.py:108
DOT_MIN = 1e-8             # renderer.py:132-135
DOT_MAX = 0.9999           # renderer.py:132-135
DET_MIN = 1e-15            # renderer.py:51
LAMBDA_INNER_MIN = 1e-15   # renderer.py:69-72
NORM_EPS = 1e-20           # data_utils.py:54  torch_norm eps (render_utils.py:11 too)
DISP_VAL = 0.01            # dataLoader.py:154 (matup_render.py:45-47)
SRGB_LIMIT = 0.04045       # file_utils.py:92
GAMMA_LIMIT = 0.0031308    # render_utils.py:118
L21_EPS = 1e-15            # train.py:19-21

# Ours, not MatUp's - see the DECODE section of the module docstring.
NZ_MIN = 1e-3
MASK_MIN_NORM = 0.1
NORMAL_NORM_FLOOR = 1e-4

# Slot order inside a (B, 5, H, W, 3) stack. Must equal grouped_dataset.MAP_KEYS
# (kept as a separate literal so this file does not import the data pipeline).
MAP_SLOTS = ("baseColor", "normal", "roughness", "metallic", "height")
_BC, _NRM, _RGH, _MTL, _HGT = 0, 1, 2, 3, 4

# The configs' `render:` block. Every key is validated and cast by
# validate_render_cfg - PyYAML reads `1e-4` (no !!float) as a STRING.
DEFAULTS = {
    "n_lights": 100,
    "light_radius": 1.4,
    "intensity": INTENSITY,
    "disp_val": DISP_VAL,
    "min_roughness": MIN_ROUGHNESS,
    "min_alpha": MIN_ALPHA,
    "nz_min": NZ_MIN,
    "mask_min_norm": MASK_MIN_NORM,
    "normal_norm_floor": NORMAL_NORM_FLOOR,
    "light_chunks": 1,
}
_INT_KEYS = ("n_lights", "light_chunks")
# What MatUp itself uses (main_optim.py:87-89,122; the renderer constants above).
# A config may change these, but never silently.
_MATUP_VALUES = {
    "n_lights": 100,
    "light_radius": 1.4,
    "intensity": INTENSITY,
    "disp_val": DISP_VAL,
    "min_roughness": MIN_ROUGHNESS,
    "min_alpha": MIN_ALPHA,
}


def validate_render_cfg(rcfg, warn=True):
    """dict (or None) -> a complete dict with every value cast and range-checked.

    Unknown keys are an error: a typo such as `light_chunk: 4` would otherwise be
    ignored and the run would OOM with the default. Idempotent, so a validated
    dict can be passed through it again.
    """
    rcfg = dict(rcfg or {})
    unknown = sorted(set(rcfg) - set(DEFAULTS))
    if unknown:
        raise ValueError("render: unknown key(s) %s; allowed: %s"
                         % (unknown, sorted(DEFAULTS)))
    out = {}
    for k, default in DEFAULTS.items():
        v = rcfg.get(k, default)
        if v is None:
            raise ValueError("render.%s is null" % k)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise ValueError("render.%s = %r is not a number" % (k, v))
        if not np.isfinite(fv):
            raise ValueError("render.%s = %r is not finite" % (k, v))
        if k in _INT_KEYS:
            if fv != int(fv):
                raise ValueError("render.%s = %r must be an integer" % (k, v))
            out[k] = int(fv)
        else:
            out[k] = fv

    if out["n_lights"] < 1:
        raise ValueError("render.n_lights must be >= 1")
    if out["light_chunks"] < 1 or out["n_lights"] % out["light_chunks"]:
        raise ValueError("render.light_chunks = %d must be >= 1 and divide n_lights = %d"
                         % (out["light_chunks"], out["n_lights"]))
    for k in ("light_radius", "intensity", "normal_norm_floor"):
        if out[k] <= 0:
            raise ValueError("render.%s must be > 0" % k)
    for k in ("min_roughness", "min_alpha"):
        if not 0 < out[k] <= 1:
            raise ValueError("render.%s must be in (0, 1]" % k)
    if not 0 < out["nz_min"] < 1:
        raise ValueError("render.nz_min must be in (0, 1)")
    if out["disp_val"] < 0 or out["mask_min_norm"] < 0:
        raise ValueError("render.disp_val and render.mask_min_norm must be >= 0")

    for k, mv in _MATUP_VALUES.items():
        if warn and out[k] != mv:
            print("[warn] render.%s = %r differs from MatUp's %r - the render term "
                  "no longer matches MatUp's renderer" % (k, out[k], mv))
    return out


# ------------------------------------------------------------ colour transforms

def srgb_to_linear(x):
    """file_utils.py:91-98 srgb2linrgb.

    The clamp inside pow keeps the unselected branch finite, so tf.where's
    gradient stays NaN-free (file_utils.py:97: "clamp to stabilize training").
    """
    return tf.where(x > SRGB_LIMIT,
                    tf.pow((tf.maximum(x, SRGB_LIMIT) + 0.055) / 1.055, 2.4),
                    x / 12.92)


def linear_to_srgb(x):
    """render_utils.py:116-125 gammaCorrection (linrgb2srgb), clamp inside pow."""
    return tf.where(x > GAMMA_LIMIT,
                    1.055 * tf.pow(tf.maximum(x, GAMMA_LIMIT), (1.0 / 2.4)) - 0.055,
                    12.92 * x)


def reinhard(t):
    """render_utils.py:108-109 reinhardTonemapper."""
    return t / (1 + t)


def process_raw_render(t):
    """render_utils.py:134-137: Reinhard per channel, then gamma."""
    return linear_to_srgb(reinhard(t))


# --------------------------------------------------------------------- geometry

def safe_normalize(v, axis):
    """data_utils.py:54-59 torch_norm: v / sqrt(max(sum(v^2), 1e-20))."""
    return tf.math.divide(v, tf.sqrt(tf.maximum(
        tf.reduce_sum(tf.square(v), axis=axis, keepdims=True), NORM_EPS)))


def branchless_onb(n):
    """render_utils.py:94-105 branchlessONB (Duff et al. 2017).

    n is (N, 3, 1), as renderer.py:137 passes it; returns T, B as (N, 3, 1),
    i.e. with renderer.py:138-139's unsqueeze(-1) already applied. After the n_z
    guard in decode_maps, sign(n_z) is always +1, so 1 / (sign + n_z) is finite.
    """
    sign = tf.sign(n[:, 2])                                         # :96   (N, 1)
    a = -1.0 / (sign + n[:, 2])                                     # :97
    b = n[:, 0] * n[:, 1] * a                                       # :98
    b1 = tf.concat([                                                # :99-101
        1.0 + sign * n[:, 0] * n[:, 0] * a,
        sign * b, -sign * n[:, 0]], axis=1)
    b2 = tf.concat([                                                # :102-104
        b, sign + n[:, 1] * n[:, 1] * a,
        -n[:, 1]], axis=1)
    return b1[..., None], b2[..., None]                             # renderer.py:138-139


# ------------------------------------------------------------------ BRDF terms

def d_ggx_anisotropic(a_sqr, b_sqr, wh, t, b, n_dot_h):
    """renderer.py:47-58 D_GGX_Anisotropic with c_sqr = 0 (renderer.py:174).

    n_dot_h is the CLAMPED dot; ToH/BoH are not clamped. Do not substitute the
    textbook isotropic form - it differs where the clamp binds.
    """
    to_h = tf.reduce_sum(wh * t, axis=1, keepdims=True)             # :49
    bo_h = tf.reduce_sum(wh * b, axis=1, keepdims=True)             # :50
    det_2 = tf.maximum(a_sqr * b_sqr, DET_MIN)                      # :51  (- c*c = -0)
    denom = tf.sqrt(det_2) * tf.square(                             # :52-56  pow(., 2.0)
        (a_sqr * to_h * to_h / det_2
         + b_sqr * bo_h * bo_h / det_2)                             # (+ 2c.. / det = +-0)
        + n_dot_h * n_dot_h)
    denom = tf.maximum(denom, NDF_DENOM_MIN)                        # :57
    return ONE_OVER_PI / denom                                      # :58


def v_smith_ggx_correlated_aniso(a_sqr, b_sqr, t, b, no_l, no_v, l, v):
    """renderer.py:60-85 V_SmithGGXCorrelated_aniso with c_sqr = 0.

    no_l/no_v are the clamped dots; the T/B projections are not. Note the
    pairing at :74-75: lambdaV multiplies N.L by the VIEW term, lambdaL
    multiplies N.V by the LIGHT term. O = 1 (one view), so :77-79's reshape is
    the identity.
    """
    to_v = tf.reduce_sum(v * t, axis=1, keepdims=True)              # :64  (N,1,1)
    to_l = tf.reduce_sum(l * t, axis=1, keepdims=True)              # :65  (N,1,L)
    bo_v = tf.reduce_sum(v * b, axis=1, keepdims=True)              # :66
    bo_l = tf.reduce_sum(l * b, axis=1, keepdims=True)              # :67

    lambda_v_v = tf.maximum(                                        # :69-70
        b_sqr * to_v * to_v + a_sqr * bo_v * bo_v + no_v * no_v, LAMBDA_INNER_MIN)
    lambda_v_l = tf.maximum(                                        # :71-72
        b_sqr * to_l * to_l + a_sqr * bo_l * bo_l + no_l * no_l, LAMBDA_INNER_MIN)

    lambda_v = no_l * tf.sqrt(lambda_v_v)                           # :74
    lambda_l = no_v * tf.sqrt(lambda_v_l)                           # :75

    lambda_v = tf.maximum(lambda_v, VIS_MIN)                        # :81
    lambda_l = tf.maximum(lambda_l, VIS_MIN)                        # :82
    return 0.5 / (lambda_v + lambda_l)                              # :83


def f_schlick(u, f0):
    """renderer.py:87-89 F_Schlick."""
    return f0 + (1.0 - f0) * tf.pow(1.0 - u, 5.0)


# ---------------------------------------------------------------------- decode

def _flat(t, c):
    return tf.reshape(t, [-1, c])


def _mean_c(t):
    """(B, H, W, 3) -> (N, 1): the 3->1 collapse of a grey map."""
    return tf.reshape(tf.reduce_mean(t, axis=-1, keepdims=True), [-1, 1])


def _unit_normal(enc, floor):
    """Encoded (B, H, W, 3) normal -> (floored unit n (N, 3), sum(n^2) (N, 1)).

    Shared by decode_maps and RenderLoss.diagnostics so the guard statistic is
    computed on exactly the vector the guard sees.
    """
    n = _flat(2.0 * enc - 1.0, 3)                                   # dataLoader.py:73
    sq = tf.reduce_sum(tf.square(n), axis=1, keepdims=True)
    return n / tf.sqrt(tf.maximum(sq, floor)), sq


def decode_maps(x5, pos_xy, rcfg=None):
    """Encoded map stack -> the flat renderer inputs (see module docstring).

    x5      (B, 5, H, W, 3) float32 in network/encoded space, slots MAP_SLOTS.
    pos_xy  (B, H, W, 2) float32: the [-1,1]^2 grid of the FULL material, cut and
            augmented with the maps (grouped_dataset.crop_and_augment).
    rcfg    a render config dict (validate_render_cfg output, or None).

    Returns a dict of tensors flattened over (B, H, W), N = B*H*W:
      albedo (N,3) linear, normal (N,3) unit in MatUp's frame, alpha (N,1),
      metallic (N,1), pos (N,3), valid (N,1) float {0,1} (meaningful for GT),
      nz_guarded (N,1) float {0,1} (texels the n_z guard moved).
    """
    c = validate_render_cfg(rcfg, warn=False)       # Python-time only; RenderLoss warned already
    x5 = tf.convert_to_tensor(x5, tf.float32)
    pos_xy = tf.convert_to_tensor(pos_xy, tf.float32)
    if x5.shape.rank is not None and (x5.shape.rank != 5 or x5.shape[1] not in (None, 5)
                                       or x5.shape[-1] not in (None, 3)):
        raise ValueError("decode_maps wants (B, 5, H, W, 3), got %s" % (x5.shape,))

    # baseColor: matup_render.py:62 (srgb2linrgb), clamp per renderer.py:104.
    albedo = srgb_to_linear(tf.clip_by_value(_flat(x5[:, _BC], 3), 0.0, 1.0))

    # normal: dataLoader.py:72-76 + data_utils.py:81, with our floor and guard.
    n, sq = _unit_normal(x5[:, _NRM], c["normal_norm_floor"])
    nz_unit = tf.stop_gradient(n[:, 2:3])
    raw_norm = tf.sqrt(tf.stop_gradient(sq))                        # |2x - 1|, never differentiated
    valid = tf.cast((raw_norm >= c["mask_min_norm"]) & (nz_unit >= c["nz_min"]), tf.float32)
    nz_guarded = tf.cast(nz_unit < c["nz_min"], tf.float32)
    n = tf.concat([n[:, 0:1],
                   -n[:, 1:2],                                      # dataLoader.py:74-75 (OpenGL)
                   tf.maximum(n[:, 2:3], c["nz_min"])], axis=1)     # the guard (ours)
    normal = safe_normalize(n, axis=1)                              # data_utils.py:81

    # roughness -> alpha: dataLoader.py:39-40,114; matup_render.py:67-74.
    rough = tf.clip_by_value(_mean_c(x5[:, _RGH]), 0.0, 1.0)
    rough = tf.maximum(rough, c["min_roughness"])
    alpha = tf.clip_by_value(tf.square(rough), c["min_alpha"], 1.0)

    # metallic: data_utils.py:79.
    metallic = tf.clip_by_value(_mean_c(x5[:, _MTL]), 0.0, 1.0)

    # height -> z, then the 3D position: dataLoader.py:115, matup_render.py:79-83.
    height = tf.clip_by_value(_mean_c(x5[:, _HGT]), 0.0, 1.0)
    pos = tf.concat([_flat(pos_xy, 2), c["disp_val"] * (height - 0.5)], axis=1)

    return {"albedo": albedo, "normal": normal, "alpha": alpha,
            "metallic": metallic, "pos": pos,
            "valid": valid, "nz_guarded": nz_guarded}


# ---------------------------------------------------------------------- render

def render_radiance(maps, lights, intensity=INTENSITY):
    """Linear radiance (N, 3, L): renderer.py torch_Render -> torch_Render_aniso
    -> torch_render_base + torch_GGX_aniso_BRDF, view fixed at (0, 0, 1).

    maps    decode_maps output (only albedo/metallic/normal/alpha/pos are read).
    lights  (L, 3) absolute light POSITIONS (matup_lights.training_lights()).
    """
    albedo, metallic = maps["albedo"], maps["metallic"]
    normal, alpha, pos = maps["normal"], maps["alpha"], maps["pos"]
    lights = tf.convert_to_tensor(lights, tf.float32)

    # --- torch_render_base (renderer.py:91-112)
    wi = tf.transpose(lights)[None]                                 # :96   (1, 3, L)
    point_to_light = wi - pos[..., None]                            # :97   (N, 3, L)
    # :98-101 light falloff: off (main.py:56 default, matup_render.py:217).
    wi_norm = safe_normalize(point_to_light, axis=1)                # :102

    base_color = tf.clip_by_value(albedo, 0.0, 1.0)                 # :104
    metallic = tf.clip_by_value(metallic, 0.0, 1.0)                 # :105

    diffuse_color = (1.0 - metallic) * base_color                   # :107
    f0 = (1 - metallic) * F0_DIELECTRIC + metallic * base_color     # :108-109
    f0 = f0[..., None]                                              # :110  (N, 3, 1)

    # --- torch_Render_aniso (renderer.py:162-182)
    wo = tf.constant([[0.0, 0.0, 1.0]], tf.float32)                 # z_vector, :11 and :31-32
    wo_norm = tf.transpose(safe_normalize(wo, axis=1))[None]        # :169  (1, 3, 1)
    normal = safe_normalize(normal, axis=1)[..., None]              # :170  (N, 3, 1)

    a_sqr = tf.square(alpha)                                        # :172  pow(roughness, 2.0)
    b_sqr = tf.square(alpha)                                        # :173
    # :174 c_sqr = 0.0 * roughness - dropped, see module docstring.

    # --- torch_GGX_aniso_BRDF (renderer.py:114-160)
    wh_norm = safe_normalize(wi_norm + wo_norm, axis=1)             # :119-122 (O = 1)

    n_dot_h = tf.reduce_sum(normal * wh_norm, axis=1, keepdims=True)   # :124  (N, 1, L)
    n_dot_v = tf.reduce_sum(normal * wo_norm, axis=1, keepdims=True)   # :125  (N, 1, 1)
    n_dot_l = tf.reduce_sum(normal * wi_norm, axis=1, keepdims=True)   # :126  (N, 1, L)
    l_dot_h = tf.reduce_sum(wi_norm * wh_norm, axis=1, keepdims=True)  # :127  (N, 1, L)

    heavyside_d = tf.where(n_dot_h <= 0.0, 0.0, 1.0)                # :129  on the RAW dot
    heavyside_g = tf.where(l_dot_h <= 0.0, 0.0, 1.0)                # :130

    n_dot_h = tf.clip_by_value(n_dot_h, DOT_MIN, DOT_MAX)           # :132
    n_dot_v = tf.clip_by_value(n_dot_v, DOT_MIN, DOT_MAX)           # :133
    n_dot_l = tf.clip_by_value(n_dot_l, DOT_MIN, DOT_MAX)           # :134
    l_dot_h = tf.clip_by_value(l_dot_h, DOT_MIN, DOT_MAX)           # :135

    tangent, bitangent = branchless_onb(normal)                     # :137-139  (N, 3, 1)

    brdf_tmp = diffuse_color[..., None] * ONE_OVER_PI               # :141  (N, 3, 1)

    a_sqr = a_sqr[..., None]                                        # :143  (N, 1, 1)
    b_sqr = b_sqr[..., None]                                        # :144

    v = heavyside_g * v_smith_ggx_correlated_aniso(                 # :147-149
        a_sqr, b_sqr, tangent, bitangent,
        n_dot_l, n_dot_v, wi_norm, wo_norm)

    d = heavyside_d * d_ggx_anisotropic(                            # :151-153
        a_sqr, b_sqr,
        wh_norm, tangent, bitangent, n_dot_h)

    f = f_schlick(l_dot_h, f0)                                      # :155
    specu_tmp = v * f * d                                           # :156

    brdf = n_dot_l * (brdf_tmp + specu_tmp)                         # :158 (repeat(1,1,O), O = 1)

    return intensity * brdf                                         # :180


def render_display(maps, lights, intensity=INTENSITY):
    """Display-space render (N, 3, L) = process_raw_render(radiance), the space
    MatUp's loss compares in (render_utils.py:62, train.py:15)."""
    return process_raw_render(render_radiance(maps, lights, intensity))


# ------------------------------------------------------------------------ loss

def _texel_light_error(pred, gt):
    """(N, 3, L) x2 -> (N, L): sqrt(max(sum_rgb d^2, 1e-15)) (train.py:15-20)."""
    return tf.sqrt(tf.maximum(
        tf.reduce_sum(tf.square(pred - gt), axis=1), L21_EPS))


def l21(pred, gt, mask=None):
    """MatUp's l2,1 (train.py:17-21): mean over (texel, light) of the RGB L2 norm.

    With a mask (N, 1) of {0,1}, the mean runs over valid texels only:
    sum(mask * e) / (max(sum(mask), 1) * L). An all-valid mask gives exactly the
    unmasked value up to summation order.
    """
    e = _texel_light_error(pred, gt)
    if mask is None:
        return tf.reduce_mean(e)
    n_l = tf.cast(tf.shape(e)[1], tf.float32)
    return tf.reduce_sum(mask * e) / (tf.maximum(tf.reduce_sum(mask), 1.0) * n_l)


def _masked_error_sum(pred, gt, mask):
    return tf.reduce_sum(mask * _texel_light_error(pred, gt))


class RenderLoss(object):
    """The render-space term of the grouped trainer.

    Usage inside the trainer's tf.function:

        gt_disp, valid = render_loss.target(hr5, pos)      # OUTSIDE the tape
        with tf.GradientTape(...) as tape:
            ...
            l = render_loss.loss(sr5, pos, gt_disp, valid)   # unweighted l2,1

    The target is rendered once, outside the tape and under stop_gradient, so
    nothing of it is retained for backprop.

    light_chunks > 1 is the memory fallback. The light axis is split into equal
    chunks and each chunk's masked error SUM is computed under
    tf.recompute_grad, which keeps only that chunk's inputs (the ~1.7 MB of
    decoded maps, its lights and its GT slice) and recomputes the chunk's
    render in the backward pass. Every argument goes in as a tensor argument;
    the wrapped function closes over Python floats only, as recompute_grad
    requires. The chunk sums are added and divided by the same denominator as
    the unchunked path, so the two agree up to summation order (gate d,
    "chunks").
    """

    def __init__(self, rcfg):
        self.cfg = validate_render_cfg(rcfg)
        self.n_lights = self.cfg["n_lights"]
        self.light_chunks = self.cfg["light_chunks"]
        self.intensity = self.cfg["intensity"]
        self.lights_np = matup_lights.training_lights(self.n_lights, self.cfg["light_radius"])
        self.lights = tf.constant(self.lights_np, tf.float32)

        intensity = self.intensity

        def chunk_error_sum(albedo, metallic, normal, alpha, pos, lights, gt, valid):
            maps = {"albedo": albedo, "metallic": metallic, "normal": normal,
                    "alpha": alpha, "pos": pos}
            return _masked_error_sum(render_display(maps, lights, intensity), gt, valid)

        self._chunk_error_sum = tf.recompute_grad(chunk_error_sum)

        # Not "[render] ...": that prefix is the trainer's monitor line, which
        # run_deeptex.ps1 parses as `[render] step=N l21=...`.
        print("[*] render loss: lights=%d radius=%.4g chunks=%d intensity=%.4g disp_val=%.4g "
              "min_roughness=%.4g min_alpha=%.4g nz_min=%.4g mask_min_norm=%.4g "
              "normal_norm_floor=%.4g"
              % (self.n_lights, self.cfg["light_radius"], self.light_chunks,
                 self.intensity, self.cfg["disp_val"], self.cfg["min_roughness"],
                 self.cfg["min_alpha"], self.cfg["nz_min"], self.cfg["mask_min_norm"],
                 self.cfg["normal_norm_floor"]))

    def decode(self, x5, pos_xy):
        return decode_maps(x5, pos_xy, self.cfg)

    def target(self, hr5, pos_xy):
        """GT render and GT validity: (gt_display (N, 3, L), valid (N, 1)).

        Call OUTSIDE the GradientTape. Both outputs are stop_gradient'ed.
        """
        maps = decode_maps(hr5, pos_xy, self.cfg)
        disp = render_display(maps, self.lights, self.intensity)
        return tf.stop_gradient(disp), tf.stop_gradient(maps["valid"])

    def loss(self, sr5, pos_xy, gt_display, valid):
        """Unweighted masked l2,1 of render(sr5) against gt_display (a scalar)."""
        maps = decode_maps(sr5, pos_xy, self.cfg)
        if self.light_chunks == 1:
            return l21(render_display(maps, self.lights, self.intensity), gt_display, valid)

        lights = tf.split(self.lights, self.light_chunks, axis=0)
        gts = tf.split(gt_display, self.light_chunks, axis=2)
        total = tf.add_n([
            self._chunk_error_sum(maps["albedo"], maps["metallic"], maps["normal"],
                                  maps["alpha"], maps["pos"], l_c, g_c, valid)
            for l_c, g_c in zip(lights, gts)])
        return total / (tf.maximum(tf.reduce_sum(valid), 1.0) * float(self.n_lights))

    def probe(self, sr5, hr5, pos_xy):
        """Unweighted masked l2,1 of an SR stack against its GT - the [render] monitor."""
        gt_display, valid = self.target(hr5, pos_xy)
        return self.loss(sr5, pos_xy, gt_display, valid)

    def diagnostics(self, sr5):
        """Scalars on the RAW generator output (no gradient).

        oor_*        fraction of values outside [0,1] before the decode clips them -
                     baseColor over all three channels (it is clipped per
                     channel), roughness/metallic/height over the channel mean
                     (the value that is clipped). The render term has zero
                     gradient there.
        sr_nz_guard  fraction of SR texels whose unit n_z < nz_min, i.e. moved by
                     the guard; their n_z gets no render gradient either.
        """
        x = tf.stop_gradient(tf.convert_to_tensor(sr5, tf.float32))

        def oor(t):
            return tf.reduce_mean(tf.cast((t < 0.0) | (t > 1.0), tf.float32))

        n, _ = _unit_normal(x[:, _NRM], self.cfg["normal_norm_floor"])
        return {
            "oor_basecolor": oor(x[:, _BC]),
            "oor_roughness": oor(_mean_c(x[:, _RGH])),
            "oor_metallic": oor(_mean_c(x[:, _MTL])),
            "oor_height": oor(_mean_c(x[:, _HGT])),
            "sr_nz_guard": tf.reduce_mean(tf.cast(n[:, 2:3] < self.cfg["nz_min"], tf.float32)),
        }
