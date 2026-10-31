#!/usr/bin/env python3
"""Compare two TFRecords record by record (deeptex container).

    python variants/tests/l2_records.py A.tfrecord B.tfrecord --json out.json

Not the file sha: an Example's feature map is serialised in whatever order the
protobuf runtime picks, so two writers of identical content can differ in bytes.
What training reads is the sequence of (feature key -> bytes) dicts, so that is
what is compared - every key, every byte string, in record order. Also reported:
the record count and the sha256 of the img_name sequence (the order, which is the
curriculum under upstream's shuffle=False), and each file's own sha256 for the
record (informational).
"""

import argparse
import hashlib
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow as tf  # noqa: E402


def records(path):
    for raw in tf.data.TFRecordDataset(path):
        ex = tf.train.Example.FromString(raw.numpy())
        feats = {}
        for k, f in ex.features.feature.items():
            kind = f.WhichOneof("kind")
            feats[k] = (kind, tuple(getattr(f, kind).value))
        yield feats


def name_of(feats):
    v = feats.get("image/img_name")
    return v[1][0] if v else b""


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    ha, hb = hashlib.sha256(), hashlib.sha256()
    n_a = n_b = 0
    mismatches = []
    ia, ib = records(args.a), records(args.b)
    while True:
        ra = next(ia, None)
        rb = next(ib, None)
        if ra is None and rb is None:
            break
        if ra is not None:
            n_a += 1
            ha.update(name_of(ra) + b"\n")
        if rb is not None:
            n_b += 1
            hb.update(name_of(rb) + b"\n")
        if ra != rb and len(mismatches) < 10:
            i = max(n_a, n_b) - 1
            if ra is None or rb is None:
                mismatches.append("record %d present in only one file" % i)
            else:
                keys = sorted(set(ra) | set(rb))
                bad = [k for k in keys if ra.get(k) != rb.get(k)]
                mismatches.append("record %d (%s vs %s): keys differ %s"
                                  % (i, name_of(ra).decode(errors="replace"),
                                     name_of(rb).decode(errors="replace"), bad))
    res = {"a": args.a, "b": args.b, "records_a": n_a, "records_b": n_b,
           "order_sha256_a": ha.hexdigest(), "order_sha256_b": hb.hexdigest(),
           "file_sha256_a": sha_file(args.a), "file_sha256_b": sha_file(args.b),
           "mismatches": mismatches,
           "same": n_a == n_b and not mismatches and ha.hexdigest() == hb.hexdigest()}
    with open(args.json, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["same"] else 1)


if __name__ == "__main__":
    main()
