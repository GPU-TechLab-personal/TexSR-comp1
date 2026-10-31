#!/usr/bin/env python3
"""Prove a TFRecord is sound before spending hours training on it.

Stage [D] of the DeepTextures data pipeline. Runs in the deeptex container.

This script exists because three separate silent-corruption modes converge on the
training loop, and none of them raises:

  modules/dataset.py:40-50   three tf.Assert ops whose RETURN VALUES ARE DISCARDED
                             (no tf.control_dependencies). An HR/LR pair that is
                             not an exact x4 does not fail the assert - it slices
                             successfully at :56-57 and trains as a MISALIGNED
                             pair, converging to something blurred and shifted.
  modules/dataset.py:12-13   decode_png(channels=3, dtype=uint8) coerces silently:
                             alpha dropped, palette expanded, grayscale tripled,
                             16-bit truncated.
  train_esrgan.py:23-25      tf.get_logger().disabled = True, so even the warnings
                             TF would have emitted are gone.

A full pass here is strictly stronger than a per-batch graph assert and costs
nothing at training time. The key move is decoding with channels=0 - decoding the
way dataset.py does, with channels=3, is precisely what hides the coercions.

Reads the TFRecord directly rather than through modules/utils.py load_dataset,
whose repeat()/shuffle make a single complete pass impossible.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.common import paths  # noqa: E402

FEAT = {"image/img_name": tf.io.FixedLenFeature([], tf.string),
        "image/hr_encoded": tf.io.FixedLenFeature([], tf.string),
        "image/lr_encoded": tf.io.FixedLenFeature([], tf.string)}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tfrecord", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--gt-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--align-sample", type=int, default=64,
                    help="recompute the x4 relation on this many records "
                         "(imresize_np is a pure-Python loop, so this is the slow check)")
    ap.add_argument("--report", default="", help="where to write verify_report.json "
                                                 "(default: beside the tfrecord)")
    args = ap.parse_args()

    if not os.path.isfile(args.tfrecord):
        raise SystemExit("no such file: %s" % args.tfrecord)

    failures, warnings = [], []
    lr_size = args.gt_size // args.scale
    names_seen = set()
    n = 0
    shapes = []

    for raw in tf.data.TFRecordDataset(args.tfrecord):
        n += 1
        try:
            ex = tf.io.parse_single_example(raw, FEAT)
        except Exception as e:  # noqa: BLE001 - want the record index in the message
            failures.append("record %d: parse failed: %s" % (n, e))
            continue
        name = ex["image/img_name"].numpy().decode("utf-8", "replace")
        if not name:
            failures.append("record %d: empty img_name" % n)
        if name in names_seen:
            failures.append("record %d: duplicate img_name %r" % (n, name))
        names_seen.add(name)

        # channels=0 = native. This is the whole point: channels=3 would hide
        # exactly the coercions we are looking for.
        try:
            hr = tf.image.decode_png(ex["image/hr_encoded"], channels=0)
            lr = tf.image.decode_png(ex["image/lr_encoded"], channels=0)
        except Exception as e:  # noqa: BLE001
            failures.append("%s: decode failed: %s" % (name, e))
            continue

        hs, ls = hr.shape.as_list(), lr.shape.as_list()
        shapes.append((name, hs, ls))
        if hr.dtype != tf.uint8 or lr.dtype != tf.uint8:
            failures.append("%s: dtype hr=%s lr=%s (expected uint8; a 16-bit PNG "
                            "would be silently truncated at dataset.py:12-13)"
                            % (name, hr.dtype.name, lr.dtype.name))
        if len(hs) != 3 or hs[2] != 3 or len(ls) != 3 or ls[2] != 3:
            failures.append("%s: channels hr=%s lr=%s (expected 3; channels=3 in "
                            "dataset.py would have coerced this silently)"
                            % (name, hs, ls))
            continue
        if hs[0] != ls[0] * args.scale or hs[1] != ls[1] * args.scale:
            failures.append("%s: not an exact x%d pair: hr=%s lr=%s"
                            % (name, args.scale, hs, ls))
        if min(hs[0], hs[1]) < args.gt_size:
            failures.append("%s: hr %dx%d smaller than gt_size %d -> the crop at "
                            "dataset.py:53-57 divides by zero"
                            % (name, hs[1], hs[0], args.gt_size))
        if min(ls[0], ls[1]) < lr_size:
            failures.append("%s: lr %dx%d smaller than gt_size/scale %d"
                            % (name, ls[1], ls[0], lr_size))

    print("records: %d" % n)
    if n == 0:
        # repeat() of an empty dataset yields nothing, the training for-loop never
        # enters, and train_esrgan.py:151 prints "training done!" in seconds.
        failures.append("0 records: training would print 'training done!' "
                        "immediately and write no checkpoint")
    if 0 < n < args.batch_size:
        failures.append("%d records < batch_size %d" % (n, args.batch_size))

    # Alignment: re-derive LR from HR and compare. This is the check the discarded
    # tf.Assert at dataset.py:43-46 was supposed to be.
    align_checked, align_max = 0, 0.0
    if args.align_sample > 0 and not any("channels" in f for f in failures):
        # Lazily, as before: only the alignment probe needs modules/ (and with it
        # the whole of modules/utils.py's import-time TF setup).
        paths.use_deeptex()
        from modules.utils import imresize_np  # noqa: E402
        step = max(1, n // args.align_sample)
        i = 0
        for raw in tf.data.TFRecordDataset(args.tfrecord):
            i += 1
            if i % step:
                continue
            ex = tf.io.parse_single_example(raw, FEAT)
            name = ex["image/img_name"].numpy().decode("utf-8", "replace")
            hr = tf.image.decode_png(ex["image/hr_encoded"], channels=0).numpy()
            lr = tf.image.decode_png(ex["image/lr_encoded"], channels=0).numpy()
            if hr.shape[0] != lr.shape[0] * args.scale:
                continue
            recomputed = np.rint(imresize_np(hr.astype(np.float64), 1.0 / args.scale))
            recomputed = recomputed.clip(0, 255).astype(np.uint8)
            d = float(np.abs(recomputed.astype(np.int32) - lr.astype(np.int32)).max())
            align_max = max(align_max, d)
            align_checked += 1
            if align_checked >= args.align_sample:
                break
        print("alignment: %d record(s) re-derived, max |lr - imresize_np(hr)| = %g"
              % (align_checked, align_max))
        if align_checked and align_max > 0:
            warnings.append("max |lr - imresize_np(hr)| = %g (expected 0 for pairs "
                            "built by texsr_make_pairs.py with --lr-kernel bicubic; "
                            "nonzero is expected with --lr-kernel area)" % align_max)

    meta_path = args.tfrecord + ".meta.json"
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("record_count") != n:
            failures.append("record_count mismatch: meta says %s, file has %d "
                            "(truncated TFRecords do not raise on open)"
                            % (meta.get("record_count"), n))
        digest = sha256(args.tfrecord)
        if meta.get("sha256") and meta["sha256"] != digest:
            failures.append("sha256 mismatch: meta %s... file %s..."
                            % (meta["sha256"][:16], digest[:16]))
    else:
        warnings.append("no %s beside the tfrecord" % os.path.basename(meta_path))

    ok = not failures
    report = {
        "tfrecord": os.path.abspath(args.tfrecord),
        "verdict": "PASS" if ok else "FAIL",
        "records": n,
        "unique_names": len(names_seen),
        "scale": args.scale,
        "gt_size": args.gt_size,
        "align_checked": align_checked,
        "align_max_abs_diff": align_max,
        "distinct_hr_shapes": sorted({tuple(s[1]) for s in shapes})[:20],
        "failures": failures,
        "warnings": warnings,
        "meta": meta,
    }
    rpath = args.report or os.path.join(os.path.dirname(os.path.abspath(args.tfrecord)),
                                        "verify_report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("-" * 60)
    for w in warnings:
        print("[warn] %s" % w)
    for f_ in failures[:40]:
        print("[FAIL] %s" % f_)
    if len(failures) > 40:
        print("[FAIL] ... and %d more" % (len(failures) - 40))
    print("verdict: %s   report -> %s" % (report["verdict"], rpath))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
