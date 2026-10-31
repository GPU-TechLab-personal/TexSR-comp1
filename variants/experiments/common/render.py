"""
MatUp's own renderer (Cook-Torrance + GGX, orthographic view along +Z, point
lights on a 1.4-radius Fibonacci hemisphere), driven from outside the submodule.

Every render in experiments/ goes through here, so "rendered with MatUp's
model" holds for the MatUp runs, the ESRGAN render loss (experiment 3) and
the render-space evaluation alike.

Tensor conventions follow MatUp internals, not the canonical material:
    albedo    linear RGB
    normal    MatUp's internal (DirectX-like) frame: canonical OpenGL with Y
              negated, exactly what dataLoader.unpack_normal(..., True) does
    alpha     roughness**2, clamped to [Renderer.min_alpha, 1]
    position  XY pixel-centre grid over [-1,1] built at the FULL map size,
              Z = 0.01 * (height - 0.5)   (dataLoader.py:154 hardcodes 0.01)
"""
import argparse

import numpy as np
import torch

from variants.experiments.common import paths
from variants.experiments.common.protocol import srgb_to_linear

paths.add_matup()
from matup import data_utils, light_utils, render_utils   # noqa: E402
from matup.renderer import Renderer                        # noqa: E402

DISP_VAL = 0.01
LIGHT_RADIUS = 1.4          # main_optim.py:87 and render_lowres.py:56
MIN_ROUGHNESS = 0.045       # dataLoader.load_roughness floor
EVAL_LIGHT = (0.276172, 0.276172, 0.920575)   # data_utils.py:111


def init(device):
    """Mandatory before any render (torch_render_base asserts is_init)."""
    if not Renderer.is_init or Renderer.render_device != device:
        Renderer.z_vector = Renderer.z_vector.reshape(1, 3).to(device)
        Renderer.init_renderer(argparse.Namespace(light_falloff=False), device)


def fib_lights(n=100, device="cpu"):
    """The exact light set MatUp renders and optimises with."""
    return (LIGHT_RADIUS * light_utils.generate_hemisphere_pts(
        "Fibonacci", sample_count=n, near_z=False)).float().to(device)


def eval_lights(n=16, device="cpu"):
    """Render-space evaluation lights: a Fibonacci set of a DIFFERENT size than
    the 100 MatUp fits, so evaluation is not on its training lights."""
    return fib_lights(n, device)


def position_grid(full_size, border=0, device="cpu"):
    g = data_utils.compute_2D_position_grid(full_size).float()
    if border:
        g = g[border:-border, border:-border]
    return g.to(device)


def maps_to_tensors(mat, device, border=0, full_size=None):
    """Canonical numpy material -> MatUp renderer tensors (H, W, C each).

    `full_size` is the size the XY grid is built at before cropping `border`
    (MatUp builds 512 and crops to 496; matup_render.py docstring)."""
    size = mat["basecolor"].shape[0]
    full = full_size or size + 2 * border
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)

    albedo = t(srgb_to_linear(mat["basecolor"]))
    metallic = t(mat["metallic"])[..., None]
    n = mat["normal"].copy()
    n[..., 1] = -n[..., 1]
    normal = t(n)
    r = np.maximum(mat["roughness"], MIN_ROUGHNESS)
    alpha = torch.clamp(t(r ** 2), min=Renderer.min_alpha, max=1.0)[..., None]
    z = t(DISP_VAL * (mat["height"] - 0.5))[..., None]
    pos2d = position_grid(full, (full - size) // 2, device)
    pos = torch.cat([pos2d, z], dim=-1)
    return [albedo, metallic, normal, alpha, pos]


def render(maps, lights, tonemap=True, chunk=10):
    """maps: [albedo, metallic, normal, alpha, pos] with any identical leading
    shape (..., C). Returns (..., 3, L). Differentiable.

    Normals are renormalised first (MatUp does the same inside torch_Render)
    and kept in the upper hemisphere, z >= 1e-3. MatUp's branchlessONB divides
    by sign(n_z) + n_z, which is 0/0 at n_z == 0: MatUp itself never gets
    there (its normals are rebuilt from slopes with z >= sqrt(0.001)), but a
    network's normal map does - a bf16 output of exactly 0.5 encodes z == 0,
    and the experiment-3 render loss went NaN in its first 10 iterations
    (gates/gate_train_smoke.py). Normals already in the hemisphere are
    unchanged, so MatUp's own maps render exactly as before.

    tonemap=True gives MatUp's display space (Reinhard then sRGB gamma) - the
    space every PNG render and every upsampler input lives in."""
    lead = maps[0].shape[:-1]
    flat = [m.reshape(-1, m.shape[-1]) for m in maps]
    flat[0] = torch.clamp(flat[0], 0.0, 1.0)
    flat[1] = torch.clamp(flat[1], 0.0, 1.0)
    n = data_utils.torch_norm(flat[2], dimToNorm=-1)
    flat[2] = data_utils.torch_norm(torch.cat([n[:, :2], n[:, 2:].clamp_min(1e-3)], dim=1), dimToNorm=-1)
    flat[3] = torch.clamp(flat[3], Renderer.min_alpha, 1.0)
    wo = Renderer.z_vector
    outs = []
    for i in range(0, lights.shape[0], chunk):
        r = Renderer.torch_Render(*flat, lights[i:i + chunk], wo, None)   # (N, 3, l)
        outs.append(render_utils.process_raw_render(r) if tonemap else r)
    out = torch.cat(outs, dim=-1)
    return out.reshape(*lead, 3, lights.shape[0])


def render_material(mat, lights, device, border=0, full_size=None, tonemap=True):
    """Canonical material -> (H, W, 3, L) numpy, no grad."""
    with torch.no_grad():
        r = render(maps_to_tensors(mat, device, border, full_size), lights, tonemap)
    return r.cpu().numpy()


def data_vector_to_tensors(v, pos):
    """MatUp's 7-channel data vector (albedo_lin3, metallic1, normal_xy2,
    alpha1) + position -> renderer tensors. Mirrors data_utils.maps_from_data."""
    return data_utils.maps_from_data(v, pos)
