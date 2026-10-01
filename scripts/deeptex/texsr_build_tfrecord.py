#!/usr/bin/env python3
"""Pack an HR/LRx4 pair directory into the TFRecord modules/dataset.py expects.

Stage [C] of the DeepTextures data pipeline. Runs in the deeptex container.

The feature schema is byte-identical to upstream's
external/DeepTextures/data/convert_train_tfrecord.py:37-43, so
modules/dataset.py:7-13 parses the output with no change at all. Everything else
about that script is replaced, for five reasons:

  :66  globs '*.png' only            -> 100 .jpg sources dropped (moot after stage
                                        [A], which renames everything to .png, but
                                        the glob is still the wrong contract)
  :68  builds the LR path, never
       checks it exists              -> open(lr,'rb') raises MID-WRITE at :82,
                                        leaving a truncated .tfrecord that :72 then
                                        refuses to overwrite on every retry
  :70  random.shuffle, unseeded      -> the record order is irreproducible, and
                                        under upstream's shuffle=False that order
                                        IS the training curriculum
  :72  exit(1) on "already exists"   -> wrapped by "except SystemExit: pass" at
                                        :93-97, so the PROCESS EXIT CODE IS 0 and
                                        an orchestrator reads the refusal as success
  ---  no output metadata            -> a truncated TFRecord does not fail on open;
                                        TFRecordDataset just ends early, and
                                        .repeat() turns that into a short silent
                                        cycle. A recorded count and hash is the
                                        only way to notice.

--hr/--lr may be given several times, paired in order. The MatSynth arm packs its
five per-map folders (scripts/deeptex/matsynth_make_pairs.py) into ONE record this
way, because the paper trains one network on all five channel types. They must go
through the single seeded shuffle below rather than be concatenated: upstream
reads with shuffle=False (train_esrgan.py:37), so five back-to-back blocks would
train on one map at a time.

--repeat, one per --hr/--lr pair, packs that pair's records k times. It is how the
MatSynth ablations change the mix without dropping data: the channel-ratio arm
writes baseColor and normal three times each (a map's share proportional to its
real channel count, 3:3:1:1:1). Copies after the first get "~r2".."~rk" appended
to img_name, because texsr_verify_tfrecord.py fails on a duplicate name, and every
copy goes through the same single shuffle as everything else. Without --repeat
the list and its permutation are exactly what they were before the option existed.
"""

import argparse
import hashlib
import json
import os
import random
import sys

import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_example(img_name, hr_bytes, lr_bytes):
    return tf.train.Example(features=tf.train.Features(feature={
        "image/img_name": _bytes_feature(img_name),
        "image/hr_encoded": _bytes_feature(hr_bytes),
        "image/lr_encoded": _bytes_feature(lr_bytes),
    }))


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
    ap.add_argument("--hr", required=True, action="append",
                    help="HR tile directory; repeat to pack several")
    ap.add_argument("--lr", required=True, action="append",
                    help="LRx4 tile directory, paired with the --hr in the same position")
    ap.add_argument("--repeat", type=int, action="append", default=None,
                    help="times to pack the --hr/--lr pair in the same position "
                         "(give one per pair, or none for all 1)")
    ap.add_argument("--out", required=True, help="output .tfrecord path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=32,
                    help="decode-verify this many records before declaring success")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--pairs-manifest", default="",
                    help="pairs_manifest.json to record alongside the output")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if len(args.hr) != len(args.lr):
        raise SystemExit("%d --hr but %d --lr; they pair up in order"
                         % (len(args.hr), len(args.lr)))
    repeats = args.repeat or [1] * len(args.hr)
    if len(repeats) != len(args.hr) or min(repeats) < 1:
        raise SystemExit("--repeat needs one value >= 1 per --hr/--lr pair, got %s for %d pair(s)"
                         % (repeats, len(args.hr)))
    for d in args.hr + args.lr:
        if not os.path.isdir(d):
            raise SystemExit("no such directory: %s" % d)

    if os.path.exists(args.out) and not args.overwrite:
        # Plain sys.exit, NOT absl's app.run wrapped in except SystemExit.
        sys.exit("%s already exists (use --overwrite)" % args.out)

    # (file name, hr_dir, lr_dir, record name) for every record to write.
    pairs = []
    for hr_dir, lr_dir, k in zip(args.hr, args.lr, repeats):
        hr_names = sorted(f for f in os.listdir(hr_dir) if f.lower().endswith(".png"))
        if not hr_names:
            raise SystemExit("no PNGs under %s" % hr_dir)

        # Validate the COMPLETE pairing before the writer is opened. Upstream
        # discovers a missing partner mid-write and leaves a truncated file behind.
        missing = [n for n in hr_names if not os.path.isfile(os.path.join(lr_dir, n))]
        if missing:
            for n in missing[:20]:
                print("[missing LR] %s" % os.path.join(lr_dir, n))
            raise SystemExit("%d HR tile(s) in %s have no LR partner" % (len(missing), hr_dir))
        extra = sorted(set(f for f in os.listdir(lr_dir) if f.lower().endswith(".png"))
                       - set(hr_names))
        if extra:
            print("[warn] %d LR tile(s) in %s with no HR partner will be ignored: %s"
                  % (len(extra), lr_dir, ", ".join(extra[:5])))
        for r in range(1, k + 1):
            suffix = "" if r == 1 else "~r%d" % r
            pairs += [(n, hr_dir, lr_dir, os.path.splitext(n)[0] + suffix) for n in hr_names]

    # img_name is the record name; texsr_verify_tfrecord.py fails on duplicates.
    names = [p[3] for p in pairs]
    if len(names) != len(set(names)):
        raise SystemExit("the same file name appears in more than one --hr directory")

    order = pairs[:]
    random.Random(args.seed).shuffle(order)

    tmp = args.out + ".partial"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = 0
    with tf.io.TFRecordWriter(tmp) as w:
        for name, hr_dir, lr_dir, stem in order:
            with open(os.path.join(hr_dir, name), "rb") as f:
                hr_b = f.read()
            with open(os.path.join(lr_dir, name), "rb") as f:
                lr_b = f.read()
            w.write(make_example(stem.encode(), hr_b, lr_b).SerializeToString())
            n += 1
            if n % 200 == 0:
                print("  %d/%d" % (n, len(order)))
    os.replace(tmp, args.out)
    print("wrote %d records -> %s" % (n, args.out))

    # Decode a sample now so a broken record costs 2 seconds, not 12,000 steps.
    feat = {"image/img_name": tf.io.FixedLenFeature([], tf.string),
            "image/hr_encoded": tf.io.FixedLenFeature([], tf.string),
            "image/lr_encoded": tf.io.FixedLenFeature([], tf.string)}
    checked = 0
    for raw in tf.data.TFRecordDataset(args.out).take(max(0, args.sample)):
        ex = tf.io.parse_single_example(raw, feat)
        hr = tf.image.decode_png(ex["image/hr_encoded"], channels=0)
        lr = tf.image.decode_png(ex["image/lr_encoded"], channels=0)
        hs, ls = hr.shape.as_list(), lr.shape.as_list()
        if hs[0] != ls[0] * args.scale or hs[1] != ls[1] * args.scale:
            raise SystemExit("sample %s is not an exact x%d pair: %s vs %s"
                             % (ex["image/img_name"].numpy(), args.scale, hs, ls))
        checked += 1
    print("decode-verified %d sample record(s)" % checked)

    meta = {
        "tfrecord": os.path.abspath(args.out),
        "record_count": n,
        "seed": args.seed,
        "scale": args.scale,
        "sha256": sha256(args.out),
        "bytes": os.path.getsize(args.out),
        # A single directory stays a string, so meta written before --hr could
        # repeat keeps its shape.
        "hr_dir": [os.path.abspath(d) for d in args.hr] if len(args.hr) > 1
                  else os.path.abspath(args.hr[0]),
        "lr_dir": [os.path.abspath(d) for d in args.lr] if len(args.lr) > 1
                  else os.path.abspath(args.lr[0]),
        "repeat": repeats,
        "pairs_manifest": os.path.abspath(args.pairs_manifest) if args.pairs_manifest else "",
        "schema": ["image/img_name", "image/hr_encoded", "image/lr_encoded"],
    }
    mpath = args.out + ".meta.json"
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2)
    print("meta -> %s (sha256 %s...)" % (mpath, meta["sha256"][:16]))


if __name__ == "__main__":
    main()
