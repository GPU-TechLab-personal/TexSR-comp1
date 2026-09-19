#!/usr/bin/env python3
"""
Render an arbitrary SVBRDF map set with MatUp's own renderer and eval light.

    docker compose run --rm matup python /workspace/scripts/matup/matup_render.py --all

MatUp writes a render only for its own output, so the LR and GT rows of the
comparison montage have no render to show. This produces them, using the same
renderer, the same shading model and the same evaluation light
([0.276172, 0.276172, 0.920575], data_utils.py:110-112) so the three are
directly comparable rather than merely similar.

Nothing in external/matup_egsr is modified; its modules are imported and driven
from outside.

Two things are easy to get wrong and both fail silently:

* **Normal convention.** MatSynth maps (LR and GT) are OpenGL and must be loaded
  with `opengl_normals=True`; MatUp's own `normal_opt.png` is already in the
  internal DirectX convention and must not be flipped again. Get it backwards
  and the render is lit from the wrong side without any error.

* **The position grid resolution.** MatUp builds the XY grid at the FULL output
  size and crops afterwards, so a 496^2 render must come from a 512^2 grid
  cropped to [8:-8] - not from a grid built at 496.

`--verify` re-renders MatUp's own output maps and compares against the
`render_opt.png` it wrote. Same code, same light, so they should agree to within
16-bit quantisation; a real difference means the map preparation here is wrong.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/workspace/external/matup_egsr")

from matup import data_utils, dataLoader, file_utils, render_utils   # noqa: E402
from matup.renderer import Renderer                                   # noqa: E402

# dataLoader.py:154 hardcodes this in the dataset constructor and never reads
# args.disp_val, so the value MatUp actually optimises against is this one.
DISP_VAL = 0.01

# data_utils.py:110-112
EVAL_LIGHT = np.array([[0.276172, 0.276172, 0.920575]])


def load_set(path, suffix, opengl_normals, floor_roughness):
    """PNG folder -> the four material tensors the renderer wants.

    `suffix` is "" for an SVBRDF/GT folder and "_opt" for a MatUp run folder.
    """
    def rd(name, multi):
        return file_utils.read_img_as_float(
            os.path.join(path, name + suffix + ".png"), multi)

    albedo = file_utils.srgb2linrgb(torch.from_numpy(rd("baseColor", True))).float()
    metallic = torch.from_numpy(rd("metallic", False)).float()[..., None]
    normal = dataLoader.unpack_normal(
        torch.from_numpy(rd("normal", True)), opengl_normals).float()

    rough = torch.from_numpy(rd("roughness", False)).float()
    if floor_roughness:
        # load_roughness (dataLoader.py:34-41) floors the raw map. roughness_opt
        # stores sqrt(alpha) rather than a raw roughness, so it must not be
        # floored - alpha = value^2 round-trips it exactly either way.
        rough = torch.clamp(rough, min=0.045)
    alpha = torch.clamp(torch.pow(rough, 2.0),
                        min=Renderer.min_alpha, max=1.0).float()[..., None]

    return albedo, metallic, normal, alpha


def position3d(height_norm, grid_size):
    """XY pixel-centre grid over [-1,1] plus the displaced height as Z."""
    pos2d = data_utils.compute_2D_position_grid(grid_size).float()
    z = (DISP_VAL * (height_norm - 0.5)).float()[..., None]
    return torch.cat([pos2d, z], dim=-1).float()


def crop(t, b):
    return t[b:-b, b:-b] if b else t


def render(maps, device):
    maps = [m.to(device) for m in maps]
    maps = data_utils.get_renorm_clamped_maps(maps)
    light = torch.from_numpy(EVAL_LIGHT).float().reshape(1, 3).to(device)
    return render_utils.batch_render_tn(maps, light)


def render_source(mat_dir, sub, device, border):
    """Render an SVBRDF/ or GT/ folder. MatSynth normals are OpenGL."""
    path = os.path.join(mat_dir, sub)
    albedo, metallic, normal, alpha = load_set(path, "", True, True)
    size = albedo.shape[0]
    h = torch.from_numpy(file_utils.read_img_as_float(
        os.path.join(path, "height.png"), False)).float()
    pos = position3d(h, size)

    # Only the HR side carries MatUp's zero border, so only it gets cropped;
    # the LR render stays at its native size, like every other LR cell.
    b = border if size > 128 else 0
    maps = [crop(albedo, b), crop(metallic, b), crop(normal, b),
            crop(alpha, b), crop(pos, b)]
    return render(maps, device)


def render_bilinear(mat_dir, device, border, hr):
    """Render MatUp's own starting point: the LR maps upsampled x4 bilinearly.

    `dataLoader.load_material` applies every transform first and only then
    resizes with INTER_LINEAR (dataLoader.py:115-128), and it upsamples the
    normal's XY before rebuilding Z - so this mirrors that order rather than
    upsampling finished 3-channel normals, otherwise it would not be the same
    initialisation the optimisation actually starts from.
    """
    path = os.path.join(mat_dir, "SVBRDF")
    albedo, metallic, normal, alpha = load_set(path, "", True, True)
    h = torch.from_numpy(file_utils.read_img_as_float(
        os.path.join(path, "height.png"), False)).float()
    z = DISP_VAL * (h - 0.5)

    up = lambda t: torch.from_numpy(cv2.resize(
        t.numpy(), (hr, hr), interpolation=cv2.INTER_LINEAR)).float()

    albedo, metallic = up(albedo), up(metallic)[..., None]
    alpha = up(alpha)[..., None]
    normal = data_utils.normal_from_data_vector(up(normal[..., :2]))
    pos2d = data_utils.compute_2D_position_grid(hr).float()
    pos = torch.cat([pos2d, up(z)[..., None]], dim=-1).float()

    maps = [crop(albedo, border), crop(metallic, border), crop(normal, border),
            crop(alpha, border), crop(pos, border)]
    return render(maps, device)


def render_matup(mat_dir, run_dir, device, border):
    """Re-render MatUp's own output maps - used by --verify."""
    albedo, metallic, normal, alpha = load_set(run_dir, "_opt", False, False)
    size = albedo.shape[0]

    # MatUp never writes a height, so the displacement has to come back from the
    # LR input the same way the dataset built it: scale first, then bilinear x4
    # (dataLoader.py:115 then :120-128).
    h_lr = file_utils.read_img_as_float(
        os.path.join(mat_dir, "SVBRDF", "height.png"), False)
    z = DISP_VAL * (h_lr - 0.5)
    z = cv2.resize(z, (size, size), interpolation=cv2.INTER_LINEAR)
    pos2d = data_utils.compute_2D_position_grid(size).float()
    pos = torch.cat([pos2d, torch.from_numpy(z).float()[..., None]], dim=-1).float()

    maps = [crop(albedo, border), crop(metallic, border), crop(normal, border),
            crop(alpha, border), crop(pos, border)]
    return render(maps, device)


def process(mat_dir, args, device):
    name = os.path.basename(mat_dir.rstrip("/"))
    runs = sorted(d for d in glob.glob(
        os.path.join(mat_dir, "matup-output-%s" % name, "*")) if os.path.isdir(d))
    if args.run:
        runs = [d for d in runs if os.path.basename(d) == args.run]
    if not runs:
        print("  %-28s SKIP (no MatUp output)" % name)
        return False
    run_dir = runs[-1]
    b = args.tile_width // 2

    hr = None
    for sub, out_name in (("SVBRDF", "render_lr.png"), ("GT", "render_gt.png")):
        img = render_source(mat_dir, sub, device, b)
        if sub == "GT":
            hr = img.shape[0] + 2 * b
        file_utils.write_image_as_png(os.path.join(run_dir, out_name),
                                      img.detach().cpu().numpy())

    img = render_bilinear(mat_dir, device, b, hr)
    file_utils.write_image_as_png(os.path.join(run_dir, "render_bilinear.png"),
                                  img.detach().cpu().numpy())

    msg = "  %-28s %s" % (name, os.path.basename(run_dir))

    if args.verify:
        mine = render_matup(mat_dir, run_dir, device, b).detach().cpu().numpy()
        theirs = cv2.imread(os.path.join(run_dir, "render_opt.png"),
                            cv2.IMREAD_UNCHANGED).astype(np.float64) / 65535.0
        theirs = theirs[b:-b, b:-b] if b else theirs
        theirs = theirs[..., ::-1]          # cv2 gives BGR; the renderer is RGB
        diff = np.abs(mine - theirs)
        msg += "   verify: max %.5f  mean %.6f" % (diff.max(), diff.mean())

    print(msg)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/runs/matsynth")
    ap.add_argument("--material")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--tile-width", type=int, default=17)
    ap.add_argument("--verify", action="store_true",
                    help="also re-render MatUp's own maps and diff against render_opt.png")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # init_renderer only reads .light_falloff, but it is mandatory: torch_render_base
    # asserts Renderer.is_init and z_vector stays the wrong shape without it.
    Renderer.init_renderer(argparse.Namespace(light_falloff=False), device)
    print("renderer on %s" % device)

    if args.material:
        dirs = [os.path.join(args.root, args.material)]
    elif args.all:
        dirs = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                      if os.path.isdir(os.path.join(d, "SVBRDF")))
    else:
        raise SystemExit("pass --material <name> or --all")

    print("rendering %d material(s)" % len(dirs))
    done = [d for d in dirs if process(d, args, device)]
    print("\n%d material(s) rendered" % len(done))


if __name__ == "__main__":
    main()
