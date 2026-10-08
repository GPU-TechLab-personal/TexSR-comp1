"""MatUp's light tables, ported to numpy so both containers can build them.

Shared by the two sides of the DeepTextures render-loss work:

  * scripts/deeptex/render_tf.py         - the training loss (deeptex container:
    TF 2.10, Python 3.8, numpy 1.23, NO torch)
  * scripts/matup/render_parity_torch.py - gate (a), which checks this table
    against MatUp's own expression (matup container: torch, Python 3.10)
  * scripts/compare/svbrdf_render_eval.py - the held-out lights (matup container)

It lives in scripts/common, outside both, because MatUp's light_utils.py:2 does
`import torch`, which the deeptex image does not have. Importing the submodule
from Python 3.8 would also be the first thing to break if MatUp ever used 3.10
syntax. So this file is numpy only and Python-3.8 clean.

WHAT MUST MATCH, and how it is checked
--------------------------------------
MatUp optimises against

    wi_tab = 1.4 * light_utils.generate_hemisphere_pts(
        "Fibonacci", sample_count=args.nb_lights, near_z=False).float()
                                                     (main_optim.py:87-89)

Method-call precedence puts `.float()` BEFORE the multiply: the float64 table is
cast to float32 first and the 1.4 is applied in float32. Doing it the other way
round (multiply in float64, then cast) moves 58 of the 100 rows by one float32
ulp. training_lights() does it MatUp's way, and gate (a) compares it to that
literal expression for EXACT equality - a tolerance of 1 ulp would hide exactly
this mistake.

Two facts that are easy to assume wrongly:

  * The table is NOT in elevation order. Rows whose z < 0.1 are pushed up by
    sqrt (light_utils.py:16-19), so row 10 (z = 0.105) is lower than rows 0-9.
    elevation_bands() therefore thresholds the unit z of every light, never an
    index range.
  * EVAL_LIGHT has norm 1.0 and is NOT scaled by 1.4 (data_utils.py:111). It is
    held out from the training table in both direction and distance, which is
    why heldout_lights() puts it first.
  * fib(n) for a different n is NOT disjoint from the training table. Burkardt's
    azimuth theta = 2 pi i2 / phi depends only on i2 = 2i - (ng - 1), not on ng,
    so fib(16) and fib(100) share their azimuth sequence, and fib(16) row 3 lands
    0.0017 deg from training row 2. heldout_lights() therefore turns fib(n) about
    z (HELDOUT_AZIMUTH_DEG) and checks MIN_SEP_DEG against every training
    direction.
"""
import numpy as np

# main_optim.py:87 and render_lowres.py:56 both use 1.4; main_optim.py:122
# defaults --nb_lights to 100.
RADIUS = 1.4
N_TRAIN = 100

# data_utils.py:110-112 (float32 there via .float()). Shape (3,); index it as
# EVAL_LIGHT[None] for the (1, 3) layout the renderer takes.
EVAL_LIGHT = np.array([0.276172, 0.276172, 0.920575], np.float32)

# elevation_bands(): unit z below sin(20 deg) is grazing, above sin(60 deg) high.
GRAZING_DEG = 20.0
HIGH_DEG = 60.0

# heldout_lights(): fib(n) is turned this far about z before scaling. A turn about
# z keeps every unit z, so elevation_bands() of the held-out set is unchanged
# (grazing 5 / mid 9 / high 3 for n=16). 43.25 deg is the best of a 0.125-deg scan
# of [0, 360) for n=16: smallest separation from any train100 direction 2.89 deg
# (EVAL_LIGHT: 3.63 deg), against 0.0017 deg unturned. No pure turn clears 3 deg -
# train100 neighbours are ~14 deg apart and dense near the horizon.
HELDOUT_AZIMUTH_DEG = 43.25
# heldout_lights() raises if any held-out direction is closer than this to a
# training direction.
MIN_SEP_DEG = 2.5


def sphere_fibonacci_grid_points(ng):
    """(ng, 3) float64 Fibonacci spiral points on the unit sphere.

    Line-for-line port of external/matup_egsr/matup/sphere_fibonacci_grid.py:38-59
    (Burkardt's code). The scalar loops are kept on purpose: numpy may evaluate
    sin/cos of a whole array with a SIMD kernel that differs in the last bit from
    the scalar call, and the table must be the one MatUp computes.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0                                  # :40

    theta = np.zeros(ng)                                              # :42-44
    sphi = np.zeros(ng)
    cphi = np.zeros(ng)

    for i in range(0, ng):                                            # :46-50
        i2 = 2 * i - (ng - 1)
        theta[i] = 2.0 * np.pi * float(i2) / phi
        sphi[i] = float(i2) / float(ng)
        cphi[i] = np.sqrt(float(ng + i2) * float(ng - i2)) / float(ng)

    xg = np.zeros((ng, 3))                                            # :52

    for i in range(0, ng):                                            # :54-57
        xg[i, 0] = cphi[i] * np.sin(theta[i])
        xg[i, 1] = cphi[i] * np.cos(theta[i])
        xg[i, 2] = sphi[i]

    return xg


def relocate_near_z(w, power=0.5):
    """Lift near-horizontal directions, then renormalise (light_utils.py:15-22).

    Rows with z < 0.1 get z ** power; every row is then divided by its norm.
    torch.pow(t, 0.5) dispatches to sqrt (both are correctly rounded), so the
    common case uses np.sqrt rather than np.power to land on the same bits.
    Works on a copy: MatUp's version assigns into its argument, which there is
    always a fresh boolean-mask copy (light_utils.py:11).
    """
    w = np.array(w, dtype=np.float64, copy=True)
    z = w[:, 2]
    lifted = np.sqrt(z) if power == 0.5 else np.power(z, power)
    w[:, 2] = np.where(z < 0.1, lifted, z)                            # :16-19
    return w / np.linalg.norm(w, axis=1, keepdims=True)               # :20-21


def generate_fibonacci(n):
    """(n, 3) float64 unit directions on the upper hemisphere (light_utils.py:7-12).

    2n sphere points, keep z > 0 (exactly n of them: i2 is odd for even 2n, so no
    point sits on the equator), relocate once. near_z=False in main_optim.py:89,
    so generate_hemisphere_pts (light_utils.py:71-75) adds no second relocation.
    """
    xg = sphere_fibonacci_grid_points(2 * n)                          # :8-10
    return relocate_near_z(xg[xg[:, 2] > 0.0])                        # :11


def training_lights(n=N_TRAIN, radius=RADIUS):
    """(n, 3) float32 light POSITIONS MatUp trains against (main_optim.py:87-89).

    Cast to float32 first, multiply in float32 - see the module docstring.
    """
    return np.float32(radius) * generate_fibonacci(n).astype(np.float32)


def min_separation_deg(lights, n_train=N_TRAIN):
    """(L,) float64: angle in degrees from each light's DIRECTION to the nearest
    direction of training_lights(n_train). Distance (1.4 vs 1.0) is ignored."""
    l = np.asarray(lights, dtype=np.float64)
    if l.ndim != 2 or l.shape[1] != 3:
        raise ValueError("lights must be (L, 3), got %s" % (l.shape,))
    t = training_lights(n_train).astype(np.float64)
    lu = l / np.linalg.norm(l, axis=1, keepdims=True)
    tu = t / np.linalg.norm(t, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((lu @ tu.T).max(axis=1), -1.0, 1.0)))


def heldout_lights(n=16, radius=RADIUS):
    """(n + 1, 3) float32: EVAL_LIGHT at index 0, then radius * fib(n) turned
    HELDOUT_AZIMUTH_DEG about z.

    Unturned, fib(n) shares its azimuths with the training table (module
    docstring) and fib(16) row 3 IS training row 2. The turn is applied to the
    float64 unit directions, then cast and scaled the way training_lights() is.
    Raises ValueError if any direction, EVAL_LIGHT included, is within
    MIN_SEP_DEG of a training direction (min_separation_deg).
    """
    a = np.radians(HELDOUT_AZIMUTH_DEG)
    c, s = np.cos(a), np.sin(a)
    turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    fib = generate_fibonacci(n) @ turn.T
    held = np.vstack([EVAL_LIGHT[None, :],
                      np.float32(radius) * fib.astype(np.float32)]).astype(np.float32)
    sep = min_separation_deg(held)
    if sep.min() < MIN_SEP_DEG:
        raise ValueError("heldout_lights(%d): light %d is %.4f deg from a training "
                         "direction (< MIN_SEP_DEG %.2f); choose another "
                         "HELDOUT_AZIMUTH_DEG for this n"
                         % (n, int(sep.argmin()), sep.min(), MIN_SEP_DEG))
    return held


def elevation_bands(lights):
    """Split light indices by elevation of the light DIRECTION.

    Returns {"grazing": idx, "mid": idx, "high": idx} (int64 arrays, ascending
    index order). Thresholds are on unit z = z / |l|, so the 1.4 radius and the
    unscaled EVAL_LIGHT are treated alike. Never use index ranges for this: the
    table is not sorted by elevation (module docstring).
    """
    l = np.asarray(lights, dtype=np.float64)
    if l.ndim != 2 or l.shape[1] != 3:
        raise ValueError("lights must be (L, 3), got %s" % (l.shape,))
    uz = l[:, 2] / np.linalg.norm(l, axis=1)
    grazing = uz < np.sin(np.radians(GRAZING_DEG))
    high = uz > np.sin(np.radians(HIGH_DEG))
    mid = ~(grazing | high)
    return {"grazing": np.nonzero(grazing)[0],
            "mid": np.nonzero(mid)[0],
            "high": np.nonzero(high)[0]}


if __name__ == "__main__":
    t = training_lights()
    b = elevation_bands(t)
    uz = t[:, 2].astype(np.float64) / np.linalg.norm(t.astype(np.float64), axis=1)
    print("[lights] training n=%d dtype=%s radius=%.4f..%.4f unit_z=%.4f..%.4f"
          % (len(t), t.dtype, np.linalg.norm(t, axis=1).min(),
             np.linalg.norm(t, axis=1).max(), uz.min(), uz.max()))
    print("[lights] bands grazing=%d mid=%d high=%d"
          % (len(b["grazing"]), len(b["mid"]), len(b["high"])))
    h = heldout_lights()
    hb = elevation_bands(h)
    hs = min_separation_deg(h)
    print("[lights] heldout n=%d bands grazing=%d mid=%d high=%d min_sep_deg=%.3f (light %d)"
          % (len(h), len(hb["grazing"]), len(hb["mid"]), len(hb["high"]),
             hs.min(), int(hs.argmin())))
