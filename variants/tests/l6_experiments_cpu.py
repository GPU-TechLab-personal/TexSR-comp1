#!/usr/bin/env python3
"""L6 (experiments), CPU layers: the old experiments/ code and variants/experiments
side by side, everything compared for EXACT equality (torch.equal / ==).

    variants/tests/run_experiments_l6.sh cpu python variants/tests/l6_experiments_cpu.py --out runs/variants_verify/experiments/<stamp>

Layers (each a list of checks, written to <out>/cpu.json):
  constants   module constants that L0 cannot see (paths, MAPS, URLs, REPLACE_CHILDREN)
  configs     the resolved config of every arm, old file vs new file: dict and the
              yaml dump train.py writes as config.yaml (key order included), also
              with the smoke gate's --set overrides
  arms        per arm config (2-0 .. 3-2):
                build      the old trainer's model block (exec'd from the old
                           train.py source) vs core/assemble.build, same seed ->
                           G and D state_dicts (keys AND values), init report,
                           gan_maps, D mode/active, D count
                keys       state_dict key set and its prefixes (net./nets.<m>./shared./ds.<m>./joint.)
                ckpt       an old-format checkpoint dict loads strictly into the new G/D
                forward    G(lr) old vs new, fixed input
                generator  Criterion.generator, PSNR stage and GAN stage, fixed RNG:
                           total, every log (keys in order), G gradients, D BN state
                d_loss     Discriminators.loss and its D gradients (when D is active)
                named      the kept odd behaviours, asserted on both sides
  matup       matup_driver on CPU (deterministic parts): init_maps, to_canonical,
              per_light_residuals, metrics.evaluate on one eval12 material
  gates       gate_configs.py and gate_protocol.py, old vs new: exit 0, ALL PASS,
              identical stdout; gate_exp4 synthetic part (seed 0): FAILS empty and
              identical stdout
Exit 0 = every check PASS.
"""
import argparse
import ast
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

OLD2 = os.path.join(REPO, "experiments", "exp2_esrgan", "configs")
OLD3 = os.path.join(REPO, "experiments", "exp3_render_loss", "configs")
NEW2 = os.path.join(REPO, "variants", "experiments", "exp2_esrgan", "configs")
NEW3 = os.path.join(REPO, "variants", "experiments", "exp3_render_loss", "configs")
ARMS = [(OLD2, NEW2, f) for f in sorted(os.listdir(OLD2))] + [(OLD3, NEW3, f) for f in sorted(os.listdir(OLD3))]
SMOKE_SET = ["data.root=/expdata/_gate_train", "data.workers=2", "train.psnr_iters=10",
             "log.every=10", "eval.every=20", "save.every=20", "train.milestones=[15]", "train.niter=20"]

RESULTS = []


def record(layer, name, ok, detail=""):
    RESULTS.append({"layer": layer, "check": name, "pass": bool(ok), "detail": detail})
    print("  %s %-10s %-58s %s" % ("PASS" if ok else "FAIL", layer, name, detail), flush=True)


def same_tensor_dict(a, b):
    """-> (ok, detail). Keys in order, then every tensor torch.equal."""
    if list(a) != list(b):
        return False, "keys differ: %s" % sorted(set(a) ^ set(b))[:6]
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    if bad:
        k = bad[0]
        return False, "%d tensors differ, first %s max|d| %.3g" % (
            len(bad), k, float((a[k].double() - b[k].double()).abs().max()))
    return True, "%d tensors equal" % len(a)


# ------------------------------------------------------------------ constants

def layer_constants():
    from experiments.common import paths as op
    from variants.experiments.common import paths as np_
    for k in ("WORKSPACE", "MATUP", "SWINIR_CKPT", "SHARDS", "EXPDATA", "MATERIALS", "SR_CACHE",
              "PRETRAINED", "RUNS", "EVAL12_META"):
        a, b = getattr(op, k), getattr(np_, k)
        record("constants", "paths.%s" % k, a == b, b)
    import experiments.exp2_esrgan.model as om
    import experiments.exp2_esrgan.train as ot
    from variants.experiments.exp2_esrgan.arms import pretrained as npre
    from variants.experiments.exp2_esrgan.core import config as ncfg, model as nm
    record("constants", "model.MAPS", om.MAPS == nm.MAPS, str(nm.MAPS))
    record("constants", "PRETRAINED_URLS", om.PRETRAINED_URLS == npre.PRETRAINED_URLS)
    record("constants", "_OLD (ESRGAN name remap)", om._OLD == npre._OLD)
    record("constants", "REPLACE_CHILDREN", ot.REPLACE_CHILDREN == ncfg.REPLACE_CHILDREN, str(ncfg.REPLACE_CHILDREN))
    # add_matup / add_swinir: same directories end up on sys.path
    np_.add_swinir()
    want = [os.path.join(op.MATUP), os.path.join(op.MATUP, "sr_modules", "SwinIR")]
    record("constants", "add_swinir puts MatUp + SwinIR on sys.path", all(p in sys.path for p in want))


# --------------------------------------------------------------------- configs

def layer_configs():
    from experiments.exp2_esrgan.train import load_config as old_load
    from variants.experiments.exp2_esrgan.core.config import load_config as new_load
    for od, nd, f in ARMS:
        for label, over in (("", []), (" +smoke --set", SMOKE_SET)):
            a, b = old_load(os.path.join(od, f), list(over)), new_load(os.path.join(nd, f), list(over))
            da, db = yaml.safe_dump(a, sort_keys=False), yaml.safe_dump(b, sort_keys=False)
            record("configs", "%s%s" % (f, label), a == b and da == db,
                   "dict %s, config.yaml %s" % ("==" if a == b else "!=", "==" if da == db else "!="))
        raw_old = open(os.path.join(od, f), "rb").read()
        raw_new = open(os.path.join(nd, f), "rb").read()
        note = "byte-identical" + ("; base ../../exp2_esrgan/configs/2-0... resolves in the new tree too"
                                   if f.startswith("3-0") else "")
        record("configs", "%s file bytes" % f, raw_old == raw_new, note if raw_old == raw_new else "differs")


# ------------------------------------------------------------------------ arms

def old_model_block():
    """The '# ---------------- model' block of the OLD train.main, as code."""
    path = os.path.join(REPO, "experiments", "exp2_esrgan", "train.py")
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = main.body
    i0 = next(i for i, s in enumerate(body) if ast.get_source_segment(src, s).startswith('mc = cfg["model"]'))
    i1 = next(i for i, s in enumerate(body) if i > i0 and ast.get_source_segment(src, s).startswith("opt_g ="))
    mod = ast.Module(body=body[i0:i1], type_ignores=[])
    return compile(mod, path + ":main[model block]", "exec"), body[i0].lineno, body[i1].lineno - 1


def build_old(cfg, dev):
    import experiments.exp2_esrgan.train as ot
    code, _, _ = old_model_block()
    ns = dict(vars(ot))
    ns.update(cfg=cfg, dev=dev)
    exec(code, ns)
    return ns["G"], ns["init_report"], ns["discs"], ns["crit"], ns["gan_maps"], ns["dmode"]


def build_new(cfg, dev):
    from variants.experiments.common import paths
    from variants.experiments.exp2_esrgan.core import assemble
    p = assemble.build(cfg, dev, paths.PRETRAINED)
    return p.G, p.init_report, p.discs, p.crit, p.gan_maps, p.dmode


def prefixes(keys):
    out = set()
    for k in keys:
        parts = k.split(".")
        out.add(parts[0] + "." + parts[1] + ".*" if parts[0] in ("nets", "ds") else parts[0] + ".*")
    return sorted(out)


def inputs(B=2, seed=1):
    from variants.experiments.exp2_esrgan.core.model import MAPS
    g = torch.Generator().manual_seed(seed)
    lr = {m: torch.rand(B, 3, 32, 32, generator=g) for m in MAPS}
    hr = {m: torch.rand(B, 3, 128, 128, generator=g) for m in MAPS}
    return lr, hr


def grads(module):
    return {n: (p.grad.clone() if p.grad is not None else torch.zeros(0)) for n, p in module.named_parameters()}


def layer_arms():
    from experiments.exp2_esrgan.train import load_config as old_load
    dev = torch.device("cpu")
    for od, nd, f in ARMS:
        arm = f.replace(".yaml", "")
        cfg = old_load(os.path.join(od, f), [])
        t0 = time.perf_counter()
        torch.manual_seed(0)
        Go, io_, Do, Co, gmo, dmo = build_old(copy.deepcopy(cfg), dev)
        torch.manual_seed(0)
        Gn, in_, Dn, Cn, gmn, dmn = build_new(copy.deepcopy(cfg), dev)

        ok, d = same_tensor_dict(Go.state_dict(), Gn.state_dict())
        record("arms", "%s build: G state_dict (seed 0)" % arm, ok, d)
        ok, d = same_tensor_dict(Do.state_dict(), Dn.state_dict())
        record("arms", "%s build: D state_dict (seed 0)" % arm, ok, d)
        record("arms", "%s build: init report" % arm, io_ == in_,
               "%s, %s tensors" % (in_["mode"], in_.get("loaded", "-")))
        nD = len([m for m in Dn.children()])
        record("arms", "%s build: gan_maps/D mode/active/#D" % arm,
               (gmo, dmo, Do.active(), len(list(Do.children()))) == (gmn, dmn, Dn.active(), nD),
               "mode %s, active %s, %d D" % (dmn, Dn.active(), sum(1 for _ in Dn.modules() if type(_).__name__ == "VGGStyleDiscriminator")))

        gk, dk = list(Gn.state_dict()), list(Dn.state_dict())
        record("arms", "%s keys: G %s" % (arm, ",".join(prefixes(gk))),
               list(Go.state_dict()) == gk, "%d keys" % len(gk))
        record("arms", "%s keys: D %s" % (arm, ",".join(prefixes(dk)) or "(none)"),
               list(Do.state_dict()) == dk, "%d keys" % len(dk))

        # an old-format checkpoint (train.py's torch.save dict) loads strictly into the new G/D
        buf = io.BytesIO()
        torch.save({"G": Go.state_dict(), "D": Do.state_dict() if Do.active() else None,
                    "iter": 1, "train_seconds": 0.0, "config": cfg}, buf)
        buf.seek(0)
        st = torch.load(buf, map_location=dev)
        from variants.experiments.exp2_esrgan.core import assemble
        mc = st["config"]["model"]
        Ge = assemble.generator(mc["arch"], mc["input"], mc.get("num_block", 23), mc.get("num_feat", 64),
                                mc.get("num_grow_ch", 32))
        r1 = Ge.load_state_dict(st["G"])
        r2 = Dn.load_state_dict(st["D"]) if st["D"] is not None else None
        record("arms", "%s ckpt: old checkpoint loads strictly" % arm,
               not r1.missing_keys and not r1.unexpected_keys and (r2 is None or not (r2.missing_keys or r2.unexpected_keys)),
               "G via evaluate's path%s" % (", D via resume" if r2 is not None else ""))

        lr, hr = inputs()
        with torch.no_grad():
            so, sn = Go(lr), Gn(lr)
        ok, d = same_tensor_dict(so, sn)
        record("arms", "%s forward: G(lr)" % arm, ok, d)

        for stage in (True, False):
            tag = "psnr" if stage else "gan"
            for G, Dm in ((Go, Do), (Gn, Dn)):
                G.zero_grad(set_to_none=True)
                for p in Dm.parameters():
                    p.requires_grad_(False)
            torch.manual_seed(7)
            sr_o = Go(lr)
            lo, lgo = Co.generator(sr_o, hr, Do, stage)
            lo.backward()
            torch.manual_seed(7)
            sr_n = Gn(lr)
            ln, lgn = Cn.generator(sr_n, hr, Dn, stage)
            ln.backward()
            ok_total = torch.equal(lo.detach(), ln.detach())
            ok_logs = list(lgo) == list(lgn) and all(torch.equal(lgo[k].detach(), lgn[k].detach()) for k in lgo)
            okg, dg = same_tensor_dict(grads(Go), grads(Gn))
            okd, dd = same_tensor_dict(Do.state_dict(), Dn.state_dict())
            record("arms", "%s generator[%s]: total, logs, dG, D state" % (arm, tag),
                   ok_total and ok_logs and okg and okd,
                   "total %.6f; %d logs %s; grads %s; D %s" % (float(ln), len(lgn), "==" if ok_logs else "!=",
                                                           "==" if okg else dg, "==" if okd else dd))
            if stage:
                psnr_logs = list(lgn)
            else:
                gan_logs = list(lgn)
            if Do.active() and not stage:
                for Dm in (Do, Dn):
                    Dm.zero_grad(set_to_none=True)
                    for p in Dm.parameters():
                        p.requires_grad_(True)
                ldo = Do.loss(sr_o, hr)
                ldo.backward()
                ldn = Dn.loss(sr_n, hr)
                ldn.backward()
                okg, dg = same_tensor_dict(grads(Do), grads(Dn))
                record("arms", "%s d_loss: Discriminators.loss, dD" % arm,
                       torch.equal(ldo.detach(), ldn.detach()) and okg,
                       "loss %.6f; grads %s" % (float(ldn), "==" if okg else dg))

        # the kept odd behaviours, named in README.md - asserted, so a "fix" shows up here
        if arm == "2-2_separate":
            n = sum(1 for _ in Dn.modules() if type(_).__name__ == "VGGStyleDiscriminator")
            ng = sum(1 for _ in Gn.modules() if type(_).__name__ == "RRDBNet")
            record("arms", "named: 2-2 has 5 G and 5 D (config names only arch)", n == 5 and ng == 5,
                   "%d RRDBNet, %d discriminators" % (ng, n))
        if arm in ("3-1_render", "3-2_render_jointD"):
            want = {"render"} | {"%s/extra_l1" % m for m in ("basecolor", "normal", "roughness", "metallic", "height")}
            record("arms", "named: %s PSNR stage runs render + extra_l1" % arm, want <= set(psnr_logs),
                   "psnr-stage terms: %s" % ", ".join(sorted({k.split("/")[-1] for k in psnr_logs})))
        if arm == "3-2_render_jointD":
            record("arms", "named: 3-2 per-map gan weights inert, joint only",
                   "joint/gan_g" in gan_logs and not any(k.endswith("/gan_g") and not k.startswith("joint")
                                                         for k in gan_logs),
                   "gan-stage adversarial: %s" % [k for k in gan_logs if "gan" in k])
        print("    (%s: %.0fs)" % (arm, time.perf_counter() - t0), flush=True)
        del Go, Gn, Do, Dn, Co, Cn


# ----------------------------------------------------------------------- matup

def layer_matup():
    import experiments.common.matup_driver as OD
    import experiments.common.metrics as OM
    from experiments.common.protocol import load_material, crop_border
    from variants.experiments import matup_driver as ND
    from variants.experiments.common import metrics as NM
    from variants.experiments.common import paths
    dev = torch.device("cpu")
    name = "acg_wood_floor_060"
    root = os.path.join(paths.SR_CACHE, "eval12", name)
    mat = os.path.join(paths.MATERIALS, "eval12", name)
    args_o = OD.matup_args(os.path.join(mat, "LR"), os.path.join(root, "radiance_LR_L100"),
                           os.path.join(root, "radiance_swinir_L100"), "/vv/unused")
    args_n = ND.matup_args(os.path.join(mat, "LR"), os.path.join(root, "radiance_LR_L100"),
                           os.path.join(root, "radiance_swinir_L100"), "/vv/unused")
    record("matup", "matup_args", vars(args_o) == vars(args_n), "%d fields" % len(vars(args_n)))
    mo, dso = OD.init_maps(args_o, dev)
    mn, dsn = ND.init_maps(args_n, dev)
    ok = all(torch.equal(a, b) for a, b in zip(mo, mn)) and len(mo) == len(mn)
    record("matup", "init_maps (%s)" % name, ok, "%d maps %s" % (len(mn), tuple(mn[0].shape)))
    lr = load_material(os.path.join(mat, "LR"))
    co = OD.to_canonical(mo, lr_height=lr["height"], full_size=512)
    cn = ND.to_canonical(mn, lr_height=lr["height"], full_size=512)
    record("matup", "to_canonical", list(co) == list(cn) and all(np.array_equal(co[k], cn[k]) for k in co))
    so, ro = OD.per_light_residuals(mo, dso, args_o, dev)
    sn, rn = ND.per_light_residuals(mn, dsn, args_n, dev)
    record("matup", "per_light_residuals", so == sn and np.array_equal(ro, rn),
           "mean %.5f, >2x median %.4f" % (sn["mean"], sn["discard_rel"]["gt_2x_median"]))
    gt = {k: crop_border(v, 8) for k, v in load_material(os.path.join(mat, "GT")).items()}
    eo = OM.evaluate(co, gt, dev, maps=["basecolor", "normal", "roughness", "metallic"], n_lights=16, full_size=512)
    en = NM.evaluate(cn, gt, dev, maps=["basecolor", "normal", "roughness", "metallic"], n_lights=16, full_size=512)
    record("matup", "metrics.evaluate (16 lights, FLIP+LPIPS)", eo == en,
           "bc PSNR %.3f, n %.3f deg, FLIP %.4f" % (en["maps"]["basecolor"]["psnr"],
                                                   en["maps"]["normal"]["angle_mean"], en["render"]["flip"]))


# ----------------------------------------------------------------------- gates

def run_gate(path):
    p = subprocess.run([sys.executable, path], capture_output=True, text=True, cwd=REPO,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    return p.returncode, p.stdout, p.stderr


def layer_gates():
    for g in ("gate_configs.py", "gate_protocol.py"):
        t0 = time.perf_counter()
        ro, so, eo = run_gate(os.path.join(REPO, "experiments", "gates", g))
        rn, sn, en = run_gate(os.path.join(REPO, "variants", "experiments", "gates", g))
        ok = rn == 0 and "ALL PASS" in sn and sn == so and ro == 0
        record("gates", "%s (new path)" % g, ok,
               "exit %d, %s, stdout %s old (%.0fs)" % (rn, "ALL PASS" if "ALL PASS" in sn else "not ALL PASS",
                                                       "==" if sn == so else "!=", time.perf_counter() - t0))
        if not ok:
            print(sn[-2000:], en[-2000:])
    import experiments.gates.gate_exp4 as og
    import variants.experiments.gates.gate_exp4 as ng
    t0 = time.perf_counter()
    outs = []
    for mod in (og, ng):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.synthetic(torch.device("cpu"))
        outs.append(buf.getvalue())
    print(outs[1], end="")
    record("gates", "gate_exp4 synthetic (seed 0, new path)", not ng.FAILS and outs[0] == outs[1],
           "FAILS %s, stdout %s old (%.0fs)" % (ng.FAILS or "[]", "==" if outs[0] == outs[1] else "!=",
                                                time.perf_counter() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", nargs="*", default=["constants", "configs", "arms", "matup", "gates"])
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    for layer in args.layers:
        print("[%s]" % layer, flush=True)
        globals()["layer_" + layer]()
    n_fail = sum(not r["pass"] for r in RESULTS)
    out = {"torch": torch.__version__, "threads": args.threads, "seconds": time.perf_counter() - t0,
           "old_model_block_lines": "experiments/exp2_esrgan/train.py:%d-%d" % old_model_block()[1:],
           "results": RESULTS}
    with open(os.path.join(args.out, "cpu.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("L6 cpu: %d checks, %d FAIL (%.0fs)" % (len(RESULTS), n_fail, out["seconds"]))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
