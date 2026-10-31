"""
실험 4: a moment-based (FFT-free) statistics term added to MatUp's Lfit + Lreg.

Idea. The l2,1 fit keeps only what the upsampled renders agree on across
lights; the rest of S's detail is discarded texel by texel. It need not be
discarded at the level of *statistics*: model the detail MatUp's MLP adds on
top of its bilinear init, d(x) = v(x) - v0(x) in MatUp's 7-D data space
(albedo_lin 3, metallic 1, normal_xy 2, alpha 1), as a stationary Gaussian
random field and ask the optimisation to reproduce its second moments.

Target moments (estimated once per material, before optimisation). Linearise
the renderer around the init: for light w at texel x,
    E_w(x) = S(R_w)(x) - R_w(v0)(x) ~= J_w(x) d(x) + eps_w(x)
with J_w(x) the 3x7 Jacobian of the display-space render and eps the light-
inconsistent part (iid, variance s). Taking second moments,
    E[E E^T] = J Sigma J^T + s I
which is LINEAR in (Sigma, s): an ordinary least-squares problem over all
sampled (texel, light) pairs, no Fourier transform anywhere. The same
equation on finite differences (dE ~= J dd) gives Sigma_g = Cov(grad d), the
lag-1 moment that carries the field's spectral slope. Sigma is projected onto
the PSD cone after solving.

Variants (the experiment's arms):
    A  per-map statistics only: Sigma block-diagonal by map
       [albedo 3x3 | metallic 1 | normal_xy 2x2 | alpha 1]
    B  + cross-map covariance: full 7x7 Sigma
Both match lag-0 and lag-1 moments.

Loss, per step, on a separate batch of P x P output patches (the MLP is a
17x17 conv, so a (P+16)^2 input tile gives P^2 outputs with neighbours - the
single texels MatUp samples have none):
    C, C_g   = sample (non-centred) second moments of d and grad d
    L_stat   = mean_{(i,j) in mask} ((C_ij - Sigma_ij) / (s_i s_j))^2  + lam_g * (same for grad)
with s_i = sqrt(Sigma_ii): correlation-normalised, so every map counts
equally regardless of its units. Total = MatUp + lam * L_stat.

Choices that are the experimenter's, flagged here rather than buried:
  * the Jacobian is a forward difference (metallic sits at the clamp 0 on most
    texels, where a central difference would halve the derivative)
  * d is assumed zero-mean (the MLP adds detail around a mean-preserving init)
  * lam defaults to 0.003. Measured (gate_exp4.py, acg_wood_floor_060, SwinIR,
    variant B): at lam=0.01 the term averaged 4.5x the l2,1 loss over the first
    epoch (0.098 vs 0.022). Sweep it (run.py --lambdas)
  * metallic is modelled by default (the literal spec: all maps); see
    structure_mask for why --stat-exclude metallic is worth an arm
"""
import numpy as np
import torch

from experiments.common import paths, render as R

paths.add_matup()
from matup import data_utils    # noqa: E402

BLOCKS = [(0, 3), (3, 4), (4, 6), (6, 7)]
LABELS = ["albedo_r", "albedo_g", "albedo_b", "metallic", "normal_x", "normal_y", "alpha"]
MAP_CHANNELS = {"basecolor": [0, 1, 2], "metallic": [3], "normal": [4, 5], "roughness": [6]}


def structure_mask(variant, dim=7, exclude=()):
    """Which Sigma entries are modelled. `exclude` drops maps entirely: no
    parameter in the fit (their share of E goes to the others / the noise) and
    no term in the loss.

    Measured on acg_wood_floor_060 / SwinIR (gate_exp4.py): with every map in,
    metallic takes the largest variance (4e-3 vs <3e-4 for all others) on a
    non-metal, because the render is steep in metallic. That pulls against
    MatUp's own metallic regulariser (Lreg). Whether to keep metallic in is an
    experimental decision - run.py --stat-exclude metallic is the other arm."""
    if variant == "B":
        m = torch.ones(dim, dim, dtype=torch.bool)
    elif variant == "A":
        m = torch.zeros(dim, dim, dtype=torch.bool)
        for a, b in BLOCKS:
            m[a:b, a:b] = True
    else:
        raise ValueError(variant)
    for name in exclude:
        for c in MAP_CHANNELS[name]:
            m[c, :] = False
            m[:, c] = False
    return m


def _render_data(v, pos, lights):
    """7-D data vectors (N,7) + positions (N,3) -> display-space render (N,3,L).

    MatUp's get_renorm_clamped_maps exits on 2-D input (it expects its own
    (B, 1, C) training batches), hence the singleton axis."""
    maps = data_utils.get_renorm_clamped_maps(data_utils.maps_from_data(v[:, None], pos[:, None]))
    return R.render(maps, lights, tonemap=True)[:, 0]


@torch.no_grad()
def _residual_and_jacobian(v, pos, sr, lights, eps):
    base = _render_data(v, pos, lights)                      # (N,3,L)
    E = sr - base
    J = []
    for k in range(v.shape[1]):
        vk = v.clone()
        vk[:, k] += eps
        J.append((_render_data(vk, pos, lights) - base) / eps)
    return E, torch.stack(J, dim=-1)                          # (N,3,L), (N,3,L,7)


def _solve(E, J, mask, ridge=1e-6):
    """Least squares for vec(Sigma) (entries allowed by mask) and s.

    E: (M,3) samples, J: (M,3,7). Uses the 6 unique entries of each 3x3 outer
    product, off-diagonals weighted sqrt(2) (Frobenius-consistent)."""
    iu = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    w = torch.tensor([1, 1, 1, 2 ** .5, 2 ** .5, 2 ** .5], device=E.device)
    pairs = [(i, j) for i in range(7) for j in range(i, 7) if mask[i, j]]
    ata = torch.zeros(len(pairs) + 1, len(pairs) + 1, device=E.device, dtype=torch.float64)
    atb = torch.zeros(len(pairs) + 1, device=E.device, dtype=torch.float64)
    for c in range(0, E.shape[0], 200_000):
        e, j = E[c:c + 200_000].double(), J[c:c + 200_000].double()
        b = torch.stack([e[:, a] * e[:, bb] for a, bb in iu], 1) * w           # (m,6)
        cols = []
        for i, jj in pairs:
            g = torch.stack([j[:, a, i] * j[:, bb, jj] + (j[:, a, jj] * j[:, bb, i] if i != jj else 0)
                             for a, bb in iu], 1)
            cols.append(g * w)
        cols.append(torch.stack([torch.full_like(e[:, 0], 1.0 if a == bb else 0.0)
                                 for a, bb in iu], 1) * w)                     # noise s * I
        A = torch.stack(cols, -1)                                               # (m,6,P+1)
        ata += torch.einsum("mkp,mkq->pq", A, A)
        atb += torch.einsum("mkp,mk->p", A, b)
    lam = ridge * torch.trace(ata) / ata.shape[0]
    theta = torch.linalg.solve(ata + lam * torch.eye(ata.shape[0], device=E.device, dtype=ata.dtype), atb)
    S = torch.zeros(7, 7, dtype=torch.float64, device=E.device)
    for (i, j), t in zip(pairs, theta[:-1]):
        S[i, j] = S[j, i] = t
    ev, V = torch.linalg.eigh(S)
    S_psd = (V * ev.clamp_min(0)) @ V.T
    return S_psd.float(), float(theta[-1].clamp_min(0)), {"eig_min_before_psd": float(ev.min())}


def estimate_targets(dataset, args, device, variant, n_pix=6000, eps=1e-3, seed=0, exclude=()):
    """Sigma, Sigma_g (7x7), noise levels and diagnostics for one material."""
    g = torch.Generator().manual_seed(seed)
    size = dataset.output_map_size
    b = args.tile_width // 2
    ys = torch.randint(b, size - b - 1, (n_pix,), generator=g)
    xs = torch.randint(b, size - b - 1, (n_pix,), generator=g)
    lights = R.fib_lights(args.nb_lights, device)
    v0 = dataset.svbrdf_init
    pos = dataset.get_position3D()
    sr = dataset.GT_renders                                   # (H,W,3,L), display space

    def at(yy, xx):
        return (v0[yy, xx].to(device), pos[yy, xx].to(device), sr[yy, xx].to(device))

    E0, J0 = _residual_and_jacobian(*at(ys, xs)[:2], at(ys, xs)[2], lights, eps)
    Ex, Jx = _residual_and_jacobian(*at(ys, xs + 1)[:2], at(ys, xs + 1)[2], lights, eps)
    Ey, Jy = _residual_and_jacobian(*at(ys + 1, xs)[:2], at(ys + 1, xs)[2], lights, eps)

    flat = lambda E, J: (E.permute(0, 2, 1).reshape(-1, 3), J.permute(0, 2, 1, 3).reshape(-1, 3, 7))
    mask = structure_mask(variant, exclude=exclude).to(device)
    S0, s0, d0 = _solve(*flat(E0, J0), mask)
    dE = torch.cat([Ex - E0, Ey - E0])
    dJ = torch.cat([0.5 * (Jx + J0), 0.5 * (Jy + J0)])
    Sg, sg, dg = _solve(*flat(dE, dJ), mask)
    info = {"variant": variant, "exclude": list(exclude), "n_pix": n_pix, "eps": eps,
            "sigma": S0.cpu().tolist(), "sigma_grad": Sg.cpu().tolist(),
            "noise": s0, "noise_grad": sg, "labels": LABELS,
            "E_rms": float(torch.sqrt((E0 ** 2).mean())), **{"lag0_" + k: v for k, v in d0.items()},
            **{"lag1_" + k: v for k, v in dg.items()}}
    return S0, Sg, info


class StatLoss:
    def __init__(self, dataset, args, device, variant, lam=0.003, lam_grad=1.0,
                 patch=8, n_patches=16, targets=None, seed=0, exclude=()):
        self.variant, self.lam, self.lam_grad = variant, lam, lam_grad
        self.patch, self.n_patches, self.tw = patch, n_patches, args.tile_width
        self.v0 = dataset.svbrdf_init.permute(2, 0, 1)[None].to(device)     # (1,7,H,W)
        S0, Sg, self.info = targets or estimate_targets(dataset, args, device, variant, exclude=exclude)
        self.mask = structure_mask(variant, exclude=exclude).to(device)
        self.S0, self.Sg = S0.to(device), Sg.to(device)
        self.n0 = self._norm(self.S0)
        self.ng = self._norm(self.Sg)
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.device = device

    @staticmethod
    def _norm(S):
        s = torch.sqrt(torch.diagonal(S).clamp_min(0))
        s = s.clamp_min(1e-4 * float(s.max().clamp_min(1e-12)))
        return s[:, None] * s[None, :]

    def _patches(self, model):
        span = self.patch + self.tw - 1
        H = self.v0.shape[-1]
        ys = torch.randint(0, H - span + 1, (self.n_patches,), device=self.device, generator=self.g)
        xs = torch.randint(0, H - span + 1, (self.n_patches,), device=self.device, generator=self.g)
        tiles = torch.cat([self.v0[..., y:y + span, x:x + span] for y, x in zip(ys.tolist(), xs.tolist())])
        return model(tiles)                                                  # (n,7,P,P): the MLP delta

    def _err(self, C, S, N):
        return (((C - S) / N) ** 2)[self.mask].mean()

    def __call__(self, model, dataset, step):
        d = self._patches(model)
        flat = d.permute(0, 2, 3, 1).reshape(-1, 7)
        C0 = flat.T @ flat / flat.shape[0]
        gx = (d[..., :, 1:] - d[..., :, :-1]).permute(0, 2, 3, 1).reshape(-1, 7)
        gy = (d[..., 1:, :] - d[..., :-1, :]).permute(0, 2, 3, 1).reshape(-1, 7)
        gg = torch.cat([gx, gy])
        Cg = gg.T @ gg / gg.shape[0]
        l0 = self._err(C0, self.S0, self.n0)
        lg = self._err(Cg, self.Sg, self.ng)
        return self.lam * (l0 + self.lam_grad * lg), {"stat_lag0": l0.detach(), "stat_lag1": lg.detach()}


def factory(variant, lam, lam_grad=1.0, **kw):
    """extra_factory for matup_driver.optimise; keeps the estimated targets
    reachable (factory.last_info) so run.py can store them with the result."""
    def make(dataset, args, device):
        loss = StatLoss(dataset, args, device, variant, lam, lam_grad, **kw)
        make.last_info = loss.info
        return loss
    make.last_info = None
    return make
