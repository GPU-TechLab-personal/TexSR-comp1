#!/usr/bin/env python3
"""Gate (e): choose w_render by matching gradient norms on trained generators.

Runs in the deeptex container (TF 2.10.1, Python 3.8), on the GPU:

    docker compose run --rm deeptex python /workspace/scripts/deeptex/render_calibrate.py \\
        --cfg  /workspace/configs/deeptex/esrgan_matsynth_render_r1.yaml \\
        --ckpt /workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k/ckpt-6 \\
        --ckpt ... (one per state) --batches 20 --seed 0 \\
        --out  /workspace/runs/deeptex/render_gate/calibration.json

run_deeptex.ps1 -Stage gate-render -Gate calibrate passes six prefixes,
matsynth_all and matsynth_chratio at 30k/35k/40k, resolved by step tensor. This
script prints w_star and writes it to the JSON; it never edits a config. The
user writes `w_render: !!float <w_star>` into esrgan_matsynth_render_r1.yaml,
whose `w_render: null` keeps deeptex_train.py from starting until then.

WHY GRADIENT NORMS. There is no published weight to borrow: MatUp optimises
l2,1 plus a metallic regulariser and nothing else (matup/train.py:13-29), and
DeepTextures has no render term at all. The defensible rule is one fixed before
any result exists and blind to the evaluation set: w* is the weight at which the
render term's gradient on G is as large as the largest term already there. Under
Adam, a term far below that barely moves the update direction and a term far
above it drowns pixel/VGG/GAN. The eval12 materials are never touched here, so
choosing w cannot leak test information.

WHY OFFSET-REMOVED. The DeepTextures recipe lets a global colour offset wander
from checkpoint to checkpoint (docs/DEEPTEX-DEVIATIONS.md 0.9, 0.10), and a
render loss penalises exactly that offset. At a drifted state the plain
render gradient is inflated by the offset, w would come out low, and once R1
removes the drift the term would be too weak where detail supervision is
needed. So the rule uses render_offrm: the render l2,1 of
    sr - stop_gradient(mean_{h,w}(sr - hr))      per image, per channel
i.e. of SR with its mean offset to HR taken out, which leaves the gradient the
detail would get. The plain (render_raw) norms and ratios are recorded as well.

WHY SIX STATES. One state is one sample of a drifting process. Six trained
states (two runs x three checkpoints) give a median that no single drift
excursion sets, and a max/min spread over 3x is flagged so a wildly
state-dependent ratio is visible before anyone trusts the number. Random init is
measured too (the VGG gradient there is huge - loss_G starts near 862 in these
runs) and recorded, but never used.

WHAT IS MEASURED. For every state and each of --batches fixed grouped batches
(load_grouped_dataset with --seed, shuffle off, so every state sees the same
crops), the global norm over G's trainable variables of d(term)/d(G) for

    pixel         w_pixel   * PixelLoss(hr, sr)
    feature       w_feature * ContentLoss(hr, sr)
    gan           w_gan     * upstream GeneratorLoss(D(hr), D(sr))   (RaGAN)
    render_raw    1 * RenderLoss.loss(sr5, ...)          (unweighted)
    render_offrm  1 * RenderLoss.loss(sr5 - offset, ...)  (unweighted)

with every line as deeptex_train.py's grouped train_step builds it: G =
RRDB_Model(input_size), D = DiscriminatorVGG128(gt_size), both called with
training=True, the upstream loss modules (--losses upstream, which R0/R1 train
with), lr5/hr5 flattened to B*5 images, the GT render taken outside the tape.
Each term gets its own tf.function and its own G forward instead of one
persistent tape: the trainer holds one render term on its tape, this would hold
two plus VGG and D, and the extra forwards cost seconds. G has no BatchNorm or
dropout (modules/models.py:79-111), so every forward computes the same SR.
D's training-mode BatchNorm updates its moving averages on each call; training
mode never reads them and nothing is saved, so the norms are unaffected.

Also recorded: cos(grad render_offrm, grad feature) and cos(grad render_raw,
grad feature) - whether the render term pushes G the way VGG already does.

    ratio_offrm = max(pixel, feature, gan) / render_offrm     per batch
    state value = median over batches
    w_star      = median over the trained states, rounded to 1 significant figure

Exit status: 0 every trained state's norms finite and its render norms > 0, 1
otherwise (the JSON is written either way, with "w_star": null on a failure; the
random-init state never decides it), 2 bad argument or output path, 3 checkpoint missing or
restoring it changed nothing, 5 grouped record missing / unverified, 6 no GPU,
8 config.
"""

import argparse
import collections
import datetime
import json
import math
import os
import sys
import time

# Before the first import from external/DeepTextures: no __pycache__ in the
# submodule (its .gitignore does not list the directory).
sys.dont_write_bytecode = True

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, os.pardir, os.pardir))  # repo root
from variants.common import paths  # noqa: E402
SUBMODULE = paths.use_deeptex()

import tensorflow as tf  # noqa: E402

from modules.models import RRDB_Model, DiscriminatorVGG128  # noqa: E402
from modules.losses import PixelLoss, ContentLoss  # noqa: E402
from modules.losses import GeneratorLoss as UpstreamGLoss  # noqa: E402
from modules.utils import load_yaml, set_memory_growth  # noqa: E402

from variants.deeptex.infer import deeptex_sr  # noqa: E402
from variants.deeptex.render_loss import grouped_dataset, render_tf, terms  # noqa: E402

DEFAULT_CFG = "/workspace/variants/deeptex/render_loss/r1.yaml"   # = configs/deeptex/esrgan_matsynth_render_r1.yaml
DEFAULT_OUT = "/workspace/runs/deeptex/render_gate/calibration.json"
EXTERNAL = ("/workspace/external", paths.EXTERNAL)

TERMS = ("pixel", "feature", "gan", "render_raw", "render_offrm")
BASE_TERMS = ("pixel", "feature", "gan")
SPREAD_FLAG = 3.0


def die(code, msg):
    print("[fatal] %s" % msg, flush=True)
    sys.exit(code)


def refuse_external(path, what):
    """The deeptex container starts in external/DeepTextures (docker/compose.yaml),
    so a relative --out would be written into the submodule."""
    p = os.path.abspath(path)
    for root in EXTERNAL:
        if p == root or p.startswith(root.rstrip("/\\") + os.sep) or p.startswith(root + "/"):
            die(2, "%s %s is under %s - outputs must go to /workspace/runs/..." % (what, p, root))
    return p


def _clean(o):
    """JSON-safe copy: numpy values to Python, non-finite floats to strings.
    Strict JSON on purpose - run_deeptex.ps1 Read-WStar parses this file with
    ConvertFrom-Json, which rejects NaN and Infinity tokens."""
    if isinstance(o, dict):
        return collections.OrderedDict((str(k), _clean(v)) for k, v in o.items())
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    return o


def _finite_or_none(x):
    return float(x) if x is not None and math.isfinite(x) else None


def summary(xs):
    """median / p10 / p90 / min / max over the finite values, plus the count of
    non-finite ones (which fail the gate)."""
    a = np.asarray(xs, np.float64)
    fin = a[np.isfinite(a)]
    out = collections.OrderedDict([("n", int(a.size)), ("nonfinite", int(a.size - fin.size))])
    if fin.size:
        out["median"] = float(np.median(fin))
        out["p10"] = float(np.percentile(fin, 10))
        out["p90"] = float(np.percentile(fin, 90))
        out["min"] = float(fin.min())
        out["max"] = float(fin.max())
    else:
        out["median"] = out["p10"] = out["p90"] = out["min"] = out["max"] = None
    return out


def round_sig1(x):
    """1 significant figure: 0.0347 -> 0.03, 2.6 -> 3.0, 0.96 -> 1.0."""
    if x is None or not math.isfinite(x) or x <= 0:
        return None
    return float("%.1g" % x)


def state_label(prefix, step):
    """/.../runs/deeptex/<arm>/checkpoints[_all]/<sub>/ckpt-N -> '<arm>/ckpt-N@<step>'.
    The step is part of the label: checkpoints_all/ and checkpoints/ number the
    same step differently, and the label keys ratio_offrm_by_state."""
    parts = os.path.normpath(prefix).replace("\\", "/").split("/")
    arm = None
    for i, p in enumerate(parts):
        if p.startswith("checkpoints") and i > 0:
            arm = parts[i - 1]
    label = "%s/%s" % (arm, parts[-1]) if arm else parts[-1]
    return label + ("@%d" % step if step is not None else "")


def _m(stats):
    """A summary's median for printing (None -> nan)."""
    v = stats["median"]
    return float("nan") if v is None else v


def check_grouped_record(cfg, skip_verify):
    """Same contract as deeptex_train.py's check_grouped_dataset: the record, its
    meta with the grouped SCHEMA, and a PASS verify report for THIS file."""
    path = cfg["train_dataset"]["path"]
    meta_path = path + ".meta.json"
    if not os.path.isfile(path) or not os.path.isfile(meta_path):
        die(5, "grouped TFRecord or its .meta.json not found: %s\n"
               "        build it: scripts\\deeptex\\run_deeptex.ps1 -Stage prep-grouped" % path)
    with open(meta_path) as f:
        meta = json.load(f)
    if list(meta.get("schema") or []) != list(grouped_dataset.SCHEMA):
        die(5, "%s schema is %s, not the grouped schema" % (meta_path, meta.get("schema")))
    if meta.get("bytes") and os.path.getsize(path) != meta["bytes"]:
        die(5, "%s is %d bytes, meta says %d (truncated?)"
               % (path, os.path.getsize(path), meta["bytes"]))
    report = os.path.join(os.path.dirname(path), "verify_report.json")
    if not os.path.isfile(report):
        if skip_verify:
            print("[warn] no verify_report.json beside %s; continuing (--skip-verify-gate)" % path)
            return meta
        die(5, "no verify_report.json beside %s; run texsr_verify_grouped.py "
               "(-Stage prep-grouped does), or pass --skip-verify-gate" % path)
    with open(report) as f:
        rep = json.load(f)
    if rep.get("verdict") != "PASS":
        die(5, "%s says %s" % (report, rep.get("verdict")))
    rmeta = rep.get("meta") or {}
    for k in ("sha256", "bytes", "record_count"):
        if rmeta.get(k) != meta.get(k):
            die(5, "%s was written for a different record: %s %r != meta %r"
                   % (report, k, rmeta.get(k), meta.get(k)))
    return meta


def snapshot(variables):
    return [v.numpy().copy() for v in variables]


def changed_fraction(before, variables):
    if not before:
        return 0.0
    return sum(not np.array_equal(b, v.numpy()) for b, v in zip(before, variables)) / float(len(before))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default=DEFAULT_CFG)
    ap.add_argument("--ckpt", action="append", default=[], metavar="PREFIX",
                    help="checkpoint prefix (.../ckpt-N), repeatable; one state each")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0,
                    help="keys the fixed grouped crops and the random-init state")
    ap.add_argument("--tfrecord", default="", help="override train_dataset.path")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--skip-verify-gate", action="store_true")
    args = ap.parse_args()

    out = refuse_external(args.out, "--out")
    if args.batches < 1:
        die(2, "--batches must be >= 1")
    prefixes = []
    for p in args.ckpt:
        if p in prefixes:
            print("[warn] --ckpt %s given twice; measured once" % p)
        else:
            prefixes.append(p)
    if not prefixes:
        die(2, "pass at least one --ckpt PREFIX (the trained states w* is read from)")
    if not os.path.isfile(args.cfg):
        die(2, "--cfg %s not found" % args.cfg)
    # -Gate overfit / smoke read w_star from this file: never leave an old one
    # behind a run that fails part-way.
    if os.path.exists(out):
        os.remove(out)

    # Before anything touches the GPU (upstream's version swallows the
    # RuntimeError if it is too late, modules/utils.py:33-35).
    set_memory_growth()
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        die(6, "no GPU visible to TensorFlow")

    cfg = load_yaml(args.cfg)
    if args.tfrecord:
        cfg["train_dataset"]["path"] = args.tfrecord
    if cfg["gt_size"] != cfg["input_size"] * cfg["scale"]:
        die(8, "gt_size %d != input_size %d * scale %d"
               % (cfg["gt_size"], cfg["input_size"], cfg["scale"]))
    if cfg["ch_size"] != 3:
        die(8, "the render term needs RGB maps; ch_size is %r" % cfg["ch_size"])
    if not isinstance(cfg.get("render"), dict):
        die(8, "%s has no render: block" % args.cfg)
    try:
        rcfg = render_tf.validate_render_cfg(cfg["render"], warn=False)
    except ValueError as e:
        die(8, str(e))
    weights = collections.OrderedDict(
        (k, float(cfg[k])) for k in ("w_pixel", "w_feature", "w_gan"))
    meta = check_grouped_record(cfg, args.skip_verify_gate)

    states = [("random_init", None, None)]
    for p in prefixes:
        prefix, step, _ = deeptex_sr.check_checkpoint("", p)       # dies 3 if incomplete
        label = state_label(prefix, step)
        while label in [s[0] for s in states]:
            label += "'"
        states.append((label, prefix, step))

    # --- the trainer's models and losses (deeptex_train.py, grouped mode) -------
    # Seeded so the recorded random-init state is reproducible; it is not R0/R1's
    # init (those seed with cfg `seed` before check_gpu's matmul) and is not used.
    # tf.keras.utils.set_random_seed, not tf.random.set_seed: Keras 2.10's unseeded
    # initializers (models.py:11-15) draw from Keras's own seed generator, which
    # only set_random_seed fixes (same reason as deeptex_train.py).
    tf.keras.utils.set_random_seed(args.seed)
    generator = RRDB_Model(cfg["input_size"], cfg["ch_size"], cfg["network_G"])
    discriminator = DiscriminatorVGG128(cfg["gt_size"], cfg["ch_size"])
    pixel_loss_fn = PixelLoss(criterion=cfg["pixel_criterion"])
    fea_loss_fn = ContentLoss(criterion=cfg["feature_criterion"])
    gen_loss_fn = UpstreamGLoss(gan_type=cfg["gan_type"])
    render_loss = render_tf.RenderLoss(rcfg)

    # --- fixed batches: the first --batches of the unshuffled record -------------
    n_b, n_m = int(cfg["batch_size"]), grouped_dataset.N_MAPS
    s_lr, s_hr = int(cfg["input_size"]), int(cfg["gt_size"])
    ds = grouped_dataset.load_grouped_dataset(cfg, shuffle=False, seed=args.seed, start_index=0)
    batches = [tuple(t for t in b) for b in ds.take(args.batches)]
    if len(batches) != args.batches:
        die(5, "the grouped record yielded %d of %d batches" % (len(batches), args.batches))
    names = grouped_dataset.parse_names(cfg["train_dataset"]["path"], limit=args.batches * n_b)
    batch_names = [names[i * n_b:(i + 1) * n_b] for i in range(args.batches)]
    print("[*] %d fixed batches of %d materials (%d images), seed %d"
          % (args.batches, n_b, n_b * n_m, args.seed), flush=True)

    spec = [tf.TensorSpec([n_b, n_m, s_lr, s_lr, 3], tf.float32),       # lr5
            tf.TensorSpec([n_b, n_m, s_hr, s_hr, 3], tf.float32),       # hr5
            tf.TensorSpec([n_b, s_hr, s_hr, 2], tf.float32)]            # pos

    terms_g = ("pixel", "feature", "gan")

    def make_term(term):
        @tf.function(input_signature=spec)
        def term_grad(lr5, hr5, pos):
            lr = tf.reshape(lr5, [n_b * n_m, s_lr, s_lr, 3])            # image b*5 + m
            hr = tf.reshape(hr5, [n_b * n_m, s_hr, s_hr, 3])
            if term.startswith("render"):
                gt_disp, valid = render_loss.target(hr5, pos)           # outside the tape
            with tf.GradientTape() as tape:
                sr = generator(lr, training=True)
                if term in terms_g:
                    value = terms.g_term(term, weights, pixel_loss_fn, fea_loss_fn,
                                         gen_loss_fn, discriminator, hr, sr)
                else:
                    sr5 = tf.reshape(sr, [n_b, n_m, s_hr, s_hr, 3])
                    if term == "render_offrm":
                        sr5 = terms.remove_offset(sr5, hr5)
                    value = terms.render_term(render_loss, sr5, pos, gt_disp, valid)
            grads = tape.gradient(value, generator.trainable_variables)
            return value, tf.linalg.global_norm(grads), grads
        return term_grad

    term_fns = collections.OrderedDict((t, make_term(t)) for t in TERMS)

    @tf.function
    def cosine(a, b):
        dot = tf.add_n([tf.reduce_sum(x * y) for x, y in zip(a, b)])
        return dot / (tf.linalg.global_norm(a) * tf.linalg.global_norm(b))

    # --- measure ------------------------------------------------------------------
    ckpt = tf.train.Checkpoint(model=generator, discriminator=discriminator)
    results = []
    t_all = time.time()
    for label, prefix, step in states:
        t0 = time.time()
        entry = collections.OrderedDict([("label", label), ("prefix", prefix), ("step", step),
                                         ("used_for_w", prefix is not None)])
        if prefix is not None:
            before_g = snapshot(generator.trainable_variables)
            before_d = snapshot(discriminator.trainable_variables)
            status = ckpt.restore(prefix)
            status.expect_partial()                    # no optimizer / step objects here
            status.assert_existing_objects_matched()
            fg = changed_fraction(before_g, generator.trainable_variables)
            fd = changed_fraction(before_d, discriminator.trainable_variables)
            if fg == 0.0 or fd == 0.0:
                die(3, "restoring %s changed %s - the checkpoint resolved but nothing was "
                       "applied" % (prefix, "the generator" if fg == 0.0 else "the discriminator"))
            entry["changed_variable_fraction"] = {"G": fg, "D": fd}

        per = collections.OrderedDict((t, []) for t in TERMS)
        vals = collections.OrderedDict((t, []) for t in TERMS)
        ratio_offrm, ratio_raw, cos_offrm, cos_raw, max_term = [], [], [], [], []
        for lr5, hr5, pos in batches:
            grads = {}
            for t, f in term_fns.items():
                v, n, g = f(lr5, hr5, pos)
                per[t].append(float(n.numpy()))
                vals[t].append(float(v.numpy()))
                if t in ("feature", "render_raw", "render_offrm"):
                    grads[t] = g
            cos_offrm.append(float(cosine(grads["render_offrm"], grads["feature"]).numpy()))
            cos_raw.append(float(cosine(grads["render_raw"], grads["feature"]).numpy()))
            del grads
            base = [per[t][-1] for t in BASE_TERMS]
            top = float(np.max(base)) if all(math.isfinite(b) for b in base) else float("nan")
            max_term.append(BASE_TERMS[int(np.argmax(base))] if math.isfinite(top) else None)
            ratio_offrm.append(top / per["render_offrm"][-1] if per["render_offrm"][-1] > 0
                               else float("inf"))
            ratio_raw.append(top / per["render_raw"][-1] if per["render_raw"][-1] > 0
                             else float("inf"))

        entry["grad_norm"] = collections.OrderedDict((t, summary(per[t])) for t in TERMS)
        entry["term_value"] = collections.OrderedDict((t, summary(vals[t])) for t in TERMS)
        entry["ratio_offrm"] = summary(ratio_offrm)
        entry["ratio_raw"] = summary(ratio_raw)
        entry["cos_offrm_feature"] = summary(cos_offrm)
        entry["cos_raw_feature"] = summary(cos_raw)
        entry["max_term_counts"] = dict(collections.Counter(m for m in max_term if m))
        entry["per_batch"] = collections.OrderedDict(
            [(t, per[t]) for t in TERMS]
            + [("ratio_offrm", ratio_offrm), ("ratio_raw", ratio_raw),
               ("cos_offrm_feature", cos_offrm), ("cos_raw_feature", cos_raw)])
        entry["finite"] = bool(all(math.isfinite(x) for t in TERMS for x in per[t])
                               and all(per["render_raw"]) and all(per["render_offrm"])
                               and all(math.isfinite(x) for x in ratio_offrm + ratio_raw))
        entry["seconds"] = round(time.time() - t0, 1)
        results.append(entry)
        g = entry["grad_norm"]
        print("[calibrate] state=%s step=%s pixel=%.4g feature=%.4g gan=%.4g render_raw=%.4g "
              "render_offrm=%.4g ratio_offrm=%.4g ratio_raw=%.4g cos_offrm_feature=%.3f%s (%.0fs)"
              % (label, step, _m(g["pixel"]), _m(g["feature"]), _m(g["gan"]),
                 _m(g["render_raw"]), _m(g["render_offrm"]), _m(entry["ratio_offrm"]),
                 _m(entry["ratio_raw"]), _m(entry["cos_offrm_feature"]),
                 "" if entry["used_for_w"] else "  [recorded, not used]",
                 entry["seconds"]), flush=True)

    # --- the rule -------------------------------------------------------------------
    used = [e for e in results if e["used_for_w"]]
    by_state = collections.OrderedDict((e["label"], e["ratio_offrm"]["median"]) for e in used)
    raw_by_state = collections.OrderedDict((e["label"], e["ratio_raw"]["median"]) for e in used)
    fin = [v for v in by_state.values() if v is not None and math.isfinite(v) and v > 0]
    fin_raw = [v for v in raw_by_state.values() if v is not None and math.isfinite(v) and v > 0]
    w_unrounded = float(np.median(fin)) if fin else None
    w_candidate = round_sig1(w_unrounded)
    spread = (max(fin) / min(fin)) if fin else None
    # Only the trained states decide: random init is recorded, never used, so a
    # zero or non-finite norm there is reported but cannot fail the gate.
    ok = all(e["finite"] for e in used) and len(fin) == len(used) and w_candidate is not None
    init_finite = [e["finite"] for e in results if not e["used_for_w"]]
    # A failed calibration writes no w_star: run_deeptex.ps1 Read-WStar (-Gate
    # overfit / smoke) reads whatever file is on disk and checks only w_star.
    w_star = w_candidate if ok else None

    report = collections.OrderedDict([
        ("pass", ok),
        ("w_star", w_star),
        ("w_star_candidate", w_candidate),
        ("w_star_unrounded", _finite_or_none(w_unrounded)),
        ("random_init_finite", bool(all(init_finite)) if init_finite else None),
        ("rule", "w_star = 1-significant-figure round of the median over the trained states "
                 "of median_batches(max(|g_pixel|, |g_feature|, |g_gan|) / |g_render_offrm|); "
                 "g = d(term)/d(G trainable variables), terms weighted as in the config, "
                 "render unweighted"),
        ("spread", _finite_or_none(spread)),
        ("spread_flag", bool(spread is not None and spread > SPREAD_FLAG)),
        ("spread_flag_threshold", SPREAD_FLAG),
        ("ratio_offrm_by_state", by_state),
        ("ratio_raw_by_state", raw_by_state),
        ("w_raw_unrounded", float(np.median(fin_raw)) if fin_raw else None),
        ("yaml", "w_render: !!float %s" % w_star if w_star is not None else None),
        ("created", datetime.datetime.utcnow().isoformat() + "Z"),
        ("script", "variants/deeptex/render_loss/gates/render_calibrate.py"),
        ("tf_version", tf.__version__),
        ("cfg", os.path.abspath(args.cfg)),
        ("tfrecord", cfg["train_dataset"]["path"]),
        ("record_sha256", meta.get("sha256")),
        ("batches", args.batches), ("batch_size", n_b), ("seed", args.seed),
        ("batch_materials", batch_names),
        ("weights", weights),
        ("render_cfg", rcfg),
        ("losses", "upstream (modules/losses.py PixelLoss, ContentLoss, GeneratorLoss)"),
        ("states", results),
        ("seconds", round(time.time() - t_all, 1))])

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".partial"
    with open(tmp, "w") as f:
        json.dump(_clean(report), f, indent=1, allow_nan=False)
    os.replace(tmp, out)

    if init_finite and not all(init_finite):
        print("[warn] the random-init state has a zero or non-finite norm or ratio (recorded, "
              "not used for w_star)", flush=True)
    print("[calibrate] w_star=%s (median %s over %d trained states, spread %s%s) -> %s"
          % ("%g" % w_star if w_star is not None else "null",
             "%.4g" % w_unrounded if w_unrounded is not None else "n/a", len(used),
             "%.2fx" % spread if spread is not None else "n/a",
             ", FLAGGED > %gx" % SPREAD_FLAG if report["spread_flag"] else "", out), flush=True)
    if w_star is not None:
        print("[calibrate] write into the R1 config:  %s" % report["yaml"], flush=True)
    if not ok:
        print("[fatal] calibration not usable: a trained state's gradient norm or ratio is "
              "non-finite or zero; w_star written as null (candidate %s, see %s)"
              % ("%g" % w_candidate if w_candidate is not None else "n/a", out), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
