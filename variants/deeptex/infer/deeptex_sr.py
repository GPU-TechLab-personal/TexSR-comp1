#!/usr/bin/env python3
"""Super-resolve a directory of images with a trained DeepTextures generator.

Runs in the deeptex container. Replaces external/DeepTextures/test.py, which
cannot be used at all: it hardcodes checkpoint_dir = './checkpoints/original'
(test.py:36) while training writes ./checkpoints/<sub_name>, and offers no flag to
change it. The donor read './checkpoints/' + cfg['sub_name']; DeepTextures replaced
that with a literal plus eight commented-out alternatives.

Directory in, PNGs out. NO METRICS - those belong to the comparison protocol and
have to be computed by the same code for both methods (scripts/common/texsr_metrics.py).

Each defect in test.py and the property here that removes it:

  test.py:36            checkpoint dir hardcoded, no flag
                        -> derived from cfg['sub_name'], --ckpt-dir overrides,
                           and the checkpoint's own step is printed and recorded so
                           an SR set can never be confused with one from a
                           different training step
  test.py:57,68,70      cv2.imread gives BGR, fed to a model trained on RGB
                        (modules/dataset.py:12-13 decodes with tf.image.decode_png)
                        -> tf.io.decode_image / tf.io.encode_png end to end; cv2
                           never appears in the inference path, so the bug cannot
                           come back by accident
  modules/utils.py:401  split_large_image writes tiles as JPEG QUALITY 95 and
                        test.py:68 reads them back as the SR input, while the
                        bicubic baseline uses the untouched in-memory array - so
                        the built-in comparison is biased against SR, and a PNG
                        input is silently routed through JPEG
                        -> tiling is entirely in-memory
  modules/utils.py:376-405 + test.py:79
                        SR canvas is (H//512)*512 but HR is (H//4)*4, so anything
                        not <=512 or an exact multiple of 512 raises ValueError
                        -> canvas is exactly (H*4, W*4), last row/column
                           edge-anchored to cover the remainder
  test.py:75            `if len(matrix_images[0]) > 1:` - an N-row x 1-column image
                        skips the join and only the top-left tile is saved
                        -> plain 2-D grid, no special case to get wrong
  modules/utils.py:64   tensor2img truncates (astype(uint8) after *255), ~0.5 LSB
                        low, about -0.1 dB
                        -> np.rint
  test.py:108-111       `except SystemExit: pass` around app.run, so "Cannot find
                        ckpt" exits 0 and an orchestrator reads it as success
                        -> argparse and plain sys.exit(n)

RRDB_Model is built with size=None so the model is genuinely dynamic
(modules/models.py:98-99 switches to tf.shape only in that case). For our 128->512
use tiling never engages; it exists so a large input cannot silently produce a
cropped or single-tile result the way test.py does.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.util import paths  # noqa: E402
from variants.util import arm_output  # noqa: E402
SUBMODULE = paths.use_deeptex()

import tensorflow as tf  # noqa: E402

from modules.models import RRDB_Model  # noqa: E402
from modules.utils import load_yaml, set_memory_growth  # noqa: E402

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def die(code, msg):
    print("[fatal] %s" % msg)
    sys.exit(code)


def check_checkpoint(ckpt_dir, explicit):
    """Resolve and structurally validate a checkpoint.

    A TF checkpoint is a DIRECTORY, not a single file, so MatUp's exact-byte-size
    check on the SwinIR .pth (run_matup.ps1:77-81) becomes: the index resolves, and
    every data shard the index names is actually present.
    """
    prefix = explicit or tf.train.latest_checkpoint(ckpt_dir)
    if not prefix:
        die(3, "no checkpoint under %s\n"
               "        train one first, or pass --ckpt-dir / --ckpt" % ckpt_dir)
    if not os.path.isfile(prefix + ".index"):
        die(3, "%s.index is missing" % prefix)
    shards = sorted(glob.glob(prefix + ".data-*-of-*"))
    if not shards:
        die(3, "%s has no .data-*-of-* shard" % prefix)
    m = re.search(r"\.data-\d+-of-(\d+)$", shards[0])
    expected = int(m.group(1)) if m else len(shards)
    if len(shards) != expected:
        die(3, "%s: %d of %d data shards present" % (prefix, len(shards), expected))
    total = sum(os.path.getsize(s) for s in shards) + os.path.getsize(prefix + ".index")
    if total < 60 * (1 << 20):
        print("[warn] checkpoint is only %.1f MB; the generator alone is ~67 MB "
              "of float32 weights" % (total / 1e6))
    step = None
    try:
        step = int(tf.train.load_checkpoint(prefix).get_tensor(
            "step/.ATTRIBUTES/VARIABLE_VALUE"))
    except Exception:  # noqa: BLE001 - a checkpoint without a step is usable
        pass
    print("[*] checkpoint %s (%d shard(s), %.1f MB, step %s)"
          % (prefix, len(shards), total / 1e6, step))
    return prefix, step, total


def tile_grid(extent, tile, overlap):
    """variants/util/arm_output.tile_grid."""
    return arm_output.tile_grid(extent, tile, overlap)


def super_resolve(model, lr, scale, tile, overlap):
    """lr: HxWx3 float32 in [0,1]. Returns (H*scale)x(W*scale)x3 float32.
    arm_output.super_resolve around this Keras model."""
    return arm_output.super_resolve(predictor(model), lr, scale, tile, overlap)


def predictor(model):
    """The model as arm_output's predict(NHWC float32) -> NHWC float32."""
    return lambda batch: model(batch, training=False).numpy()


def to_png_bytes(arr_float):
    """rint, not truncate. modules/utils.py:64 truncates and loses ~0.5 LSB."""
    u8 = np.rint(np.clip(arr_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    return tf.io.encode_png(tf.convert_to_tensor(u8)).numpy(), u8


def read_rgb(path):
    """The exact decoder training used: tf.image.decode_png via decode_image."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    return img.numpy()


def self_test(model, scale):
    """Prove the whole path in seconds: finite, in range, and exactly scale x."""
    ok = True
    ramp = np.tile(np.linspace(0, 1, 64, dtype=np.float32)[None, :, None], (64, 1, 3))
    for name, lr in (("gradient64", ramp),
                     ("noise32", np.random.RandomState(0).rand(32, 32, 3).astype(np.float32))):
        sr = super_resolve(model, lr, scale, 0, 0)
        shape_ok = sr.shape[:2] == (lr.shape[0] * scale, lr.shape[1] * scale)
        finite = bool(np.isfinite(sr).all())
        print("  %-12s %s -> %s  shape=%s finite=%s range=[%.3f, %.3f]"
              % (name, lr.shape[:2], sr.shape[:2], "OK" if shape_ok else "BAD",
                 finite, float(sr.min()), float(sr.max())))
        ok = ok and shape_ok and finite
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--ckpt", default="", help="explicit checkpoint prefix")
    ap.add_argument("--in-dir", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--make-lr", default="none", choices=["none", "bicubic"],
                    help="none = inputs are already LR")
    ap.add_argument("--tile-size", type=int, default=0,
                    help="LR-space tile; 0 = whole image (the model is dynamic)")
    ap.add_argument("--overlap", type=int, default=16, help="LR-space tile overlap")
    ap.add_argument("--save-bicubic", default="",
                    help="also write a bicubic xscale baseline here")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_memory_growth()
    cfg = load_yaml(args.cfg)
    scale = cfg["scale"]

    # size=None: the dynamic model. Building it at cfg['input_size'] would pin the
    # input to 32x32 and reject every real image (models.py:92,98-99).
    model = RRDB_Model(None, cfg["ch_size"], cfg["network_G"])

    ckpt_dir = args.ckpt_dir or os.path.join(
        cfg.get("checkpoint_root", "/workspace/runs/deeptex/checkpoints"), cfg["sub_name"])
    prefix, step, _ = check_checkpoint(ckpt_dir, args.ckpt)
    before = [v.numpy().copy() for v in model.trainable_variables]
    # expect_partial: the training checkpoint also holds optimizers and, for the
    # GAN stage, a discriminator. None of that belongs to inference.
    status = tf.train.Checkpoint(model=model).restore(prefix)
    status.expect_partial()
    # ...but every variable THIS model does have must have been matched. Upstream
    # discards the status object entirely (test.py:49, train_esrgan.py:66,73), so a
    # structurally mismatched checkpoint restores partially and silently.
    status.assert_existing_objects_matched()
    after = [v.numpy() for v in model.trainable_variables]
    unchanged = sum(1 for b, a in zip(before, after) if np.array_equal(b, a))
    if unchanged == len(before):
        die(3, "restoring %s did not change ANY of the generator's %d weight "
               "tensors - the checkpoint resolved but nothing was applied"
               % (prefix, len(before)))
    if unchanged:
        # Biases initialised to zeros can legitimately still be zeros early in
        # training, so this is a note rather than a failure.
        print("[note] %d/%d weight tensors are bit-identical to their "
              "initialisation" % (unchanged, len(before)))

    if args.self_test:
        print("[*] self-test")
        if not self_test(model, scale):
            die(3, "self-test failed")
        print("[*] self-test OK")
        if not args.in_dir:
            return

    if not args.in_dir or not args.out_dir:
        die(2, "--in-dir and --out-dir are required (unless only --self-test)")
    if not os.path.isdir(args.in_dir):
        die(2, "no such directory: %s" % args.in_dir)
    names = sorted(f for f in os.listdir(args.in_dir) if f.lower().endswith(EXTS))
    if not names:
        die(2, "no images under %s" % args.in_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_bicubic:
        os.makedirs(args.save_bicubic, exist_ok=True)

    records, failed = [], []
    for n in names:
        src = os.path.join(args.in_dir, n)
        stem = os.path.splitext(n)[0]
        try:
            img = read_rgb(src)
            if args.make_lr == "bicubic":
                h, w = img.shape[0] // scale * scale, img.shape[1] // scale * scale
                img = tf.image.resize(img[:h, :w], (h // scale, w // scale),
                                      method="bicubic", antialias=True).numpy()
                img = np.rint(np.clip(img, 0, 255)).astype(np.uint8)
            lr = img.astype(np.float32) / 255.0
            sr = super_resolve(model, lr, scale, args.tile_size, args.overlap)
            if not np.isfinite(sr).all():
                raise RuntimeError("non-finite output")
            png, _ = to_png_bytes(sr)
            out = os.path.join(args.out_dir, stem + ".png")
            with open(out, "wb") as f:
                f.write(png)
            if args.save_bicubic:
                bic = tf.image.resize(tf.convert_to_tensor(lr),
                                      (lr.shape[0] * scale, lr.shape[1] * scale),
                                      method="bicubic").numpy()
                bpng, _ = to_png_bytes(bic)
                with open(os.path.join(args.save_bicubic, stem + ".png"), "wb") as f:
                    f.write(bpng)
            records.append({"src": n, "out": stem + ".png",
                            "lr_shape": list(lr.shape), "sr_shape": list(sr.shape)})
            print("  %-28s %s -> %s" % (n, lr.shape[:2], sr.shape[:2]))
        except tf.errors.ResourceExhaustedError as e:
            # Not a per-image condition: the next image will OOM too.
            die(10, "OOM on %s - lower --tile-size\n%s" % (n, e))
        except Exception as e:  # noqa: BLE001 - one bad input must not kill the run
            msg = str(e).splitlines()[0]
            if any(s in msg for s in ("CUDA_ERROR_", "cuDNN launch failure",
                                      "illegal memory access")):
                die(11, "GPU lost on %s: %s" % (n, msg))
            failed.append((n, msg))
            print("  %-28s FAILED: %s" % (n, msg))

    manifest = {
        "checkpoint": prefix, "step": step, "cfg": os.path.abspath(args.cfg),
        "sub_name": cfg["sub_name"], "scale": scale,
        "tile_size": args.tile_size, "overlap": args.overlap,
        "make_lr": args.make_lr,
        "in_dir": os.path.abspath(args.in_dir), "out_dir": os.path.abspath(args.out_dir),
        "count": len(records), "failed": [{"src": n, "why": w} for n, w in failed],
        "files": records,
    }
    mpath = os.path.join(args.out_dir, "sr_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print("%d/%d written -> %s (manifest %s)" % (len(records), len(names), args.out_dir, mpath))
    if failed:
        sys.exit("%d image(s) failed" % len(failed))


if __name__ == "__main__":
    main()
