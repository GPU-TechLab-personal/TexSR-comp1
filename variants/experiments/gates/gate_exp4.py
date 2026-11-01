#!/usr/bin/env python3
"""
Gate for experiment 4 (statistics term). ~3 min.

    python variants/experiments/gates/gate_exp4.py

1. The moment estimator recovers a KNOWN covariance: draw d ~ N(0, Sigma_true)
   (full 7x7, cross-map terms included), synthesise E = J d + eps with random
   Jacobians, solve. Variant B must recover Sigma_true; variant A must recover
   its block diagonal (the cross-map part it is not allowed to model).
2. On a real material (eval12, S = SwinIR): targets are PSD, finite, and the
   A/B diagonals agree roughly (same data, B only adds off-block terms).
3. One optimisation epoch with variant B runs and the stat term moves.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.experiments.common import cli                               # noqa: E402
from variants.experiments.matup_grf_stats import grf_stats as G            # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print("  %-56s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILS.append(name)


def synthetic(dev):
    g = torch.Generator(device="cpu").manual_seed(0)
    A = torch.randn(7, 7, generator=g) * 0.05
    S_true = (A @ A.T + torch.diag(torch.rand(7, generator=g) * 1e-3)).to(dev)
    M = 400_000
    J = (torch.randn(M, 3, 7, generator=g) * 0.5).to(dev)
    L = torch.linalg.cholesky(S_true)
    d = (torch.randn(M, 7, generator=g).to(dev) @ L.T)
    s = 4e-3                  # light-inconsistent part comparable to the signal, as in real SR renders
    E = torch.einsum("mij,mj->mi", J, d) + torch.randn(M, 3, generator=g).to(dev) * s ** 0.5
    for variant in ("B", "A"):
        mask = G.structure_mask(variant).to(dev)
        S_hat, s_hat, _ = G._solve(E, J, mask)
        want = torch.where(mask, S_true, torch.zeros_like(S_true))
        rel = float(torch.linalg.norm(S_hat - want) / torch.linalg.norm(want))
        check("synthetic: variant %s recovers its Sigma" % variant, rel < 0.05, "rel err %.3f" % rel)
        if variant == "B":
            check("synthetic: noise level recovered", abs(s_hat - s) / s < 0.2, "%.2e vs %.2e" % (s_hat, s))


def real(dev):
    from variants.experiments import matup_driver as D
    from variants.experiments.common import paths
    from variants.experiments.common.sr_cache import ensure_sr_renders
    from variants.experiments.common.upscalers import get
    import os
    name = "acg_wood_floor_060"
    mat = os.path.join(paths.MATERIALS, "eval12", name)
    up = get("swinir", dev)
    lr_r, sr = ensure_sr_renders(os.path.join(mat, "LR"), os.path.join(paths.SR_CACHE, "eval12", name), up, dev)
    args = D.matup_args(os.path.join(mat, "LR"), lr_r, sr, "/tmp/gate_exp4/", nb_epochs=1)
    D.Renderer.init_renderer(args, dev)
    ds = D.MatUpDataset(args, dev)
    out = {}
    for v in ("A", "B"):
        S0, Sg, info = G.estimate_targets(ds, args, dev, v, n_pix=3000)
        ev = torch.linalg.eigvalsh(S0.double())
        check("real: variant %s Sigma finite and PSD" % v, bool(torch.isfinite(S0).all()) and float(ev.min()) > -1e-9,
              "min eig %.2e, E_rms %.4f" % (float(ev.min()), info["E_rms"]))
        out[v] = (S0, Sg, info)
        print("      diag(Sigma)  " + " ".join("%.2e" % x for x in torch.diagonal(S0).tolist()))
        print("      diag(Sig_g)  " + " ".join("%.2e" % x for x in torch.diagonal(Sg).tolist()))
    dA, dB = torch.diagonal(out["A"][0]), torch.diagonal(out["B"][0])
    rel = float(torch.linalg.norm(dA - dB) / torch.linalg.norm(dB).clamp_min(1e-12))
    check("real: A and B diagonals agree (rel < 0.5)", rel < 0.5, "%.3f" % rel)
    B = out["B"][0]
    s = torch.sqrt(torch.diagonal(B).clamp_min(1e-12))
    corr = B / (s[:, None] * s[None, :])
    print("      B correlation albedo_g~normal_x %.2f  albedo_g~alpha %.2f  normal_x~normal_y %.2f"
          % (corr[1, 4], corr[1, 6], corr[4, 5]))

    f = G.factory("B", 0.01, targets=out["B"])
    maps, _, _, hist = D.optimise(args, dev, f)
    check("real: 1 epoch with variant B runs", len(hist) == 1 and hist[0]["loss_extra"] > 0,
          "matup %.5f  extra %.5f  lag0 %.3f  lag1 %.3f" % (hist[0]["loss_matup"], hist[0]["loss_extra"],
                                                           hist[0].get("stat_lag0", -1), hist[0].get("stat_lag1", -1)))


def main():
    dev = cli.device()
    print("estimator on synthetic data")
    synthetic(dev)
    print("real material")
    real(dev)
    print("\n%s" % ("ALL PASS" if not FAILS else "FAILED: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
