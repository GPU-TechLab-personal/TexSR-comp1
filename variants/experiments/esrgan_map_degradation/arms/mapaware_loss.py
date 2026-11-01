"""Arm 2-3 (configs/2-3_mapaware_loss.yaml): a loss per map. normal -> angular
error only; roughness, metallic -> L1 only; basecolor, height keep ESRGAN's.

Two pieces of code make the arm:
  * AngularTerm below - the only loss term the arm adds. It joins the pixel
    group, after L1, and like L1 has weight 1.0 in the PSNR warm-up.
  * the REPLACE semantics of core/config.py (REPLACE_CHILDREN): the child's
    `normal: {angular: 1.0}` replaces 2-0's {l1, vgg, gan} instead of merging
    into it. Without it 2-3 would silently keep the base losses beside angular.
"""
import torch
import torch.nn.functional as F


def angular(pred_enc, gt_enc):
    p = F.normalize(pred_enc.float() * 2 - 1, dim=1)
    g = F.normalize(gt_enc.float() * 2 - 1, dim=1)
    cross = torch.linalg.norm(torch.cross(p, g, dim=1), dim=1)
    dot = (p * g).sum(1)
    return torch.atan2(cross, dot).mean()


def applies(per_map):
    return any("angular" in spec for spec in per_map.values())


class AngularTerm:
    """Pixel group: mean angle between decoded unit normals, radians."""
    key = "angular"

    def __call__(self, m, spec, sr, hr, psnr_stage):
        if "angular" not in spec:
            return None
        if m != "normal":
            raise ValueError("angular loss only makes sense for normal maps")
        v = angular(sr, hr)
        w = 1.0 if psnr_stage else spec["angular"]
        return w, v
