#!/usr/bin/env python3
"""L1 fixtures: small copies of the real training records, plus test configs.

Runs in the deeptex container. Takes the first N serialized records of
runs/deeptex/datasets/{textures_db,matsynth_grouped}/*.tfrecord byte for byte
(no re-encoding), writes each beside a .meta.json and a PASS verify_report.json
whose meta matches the copy (so the trainers' verify gates run, not get
skipped), copies a few held-out tiles for --val, and derives CPU-sized configs
from the real ones (batch and network depth reduced, everything else as is).

Output: runs/variants_verify/l1/fixtures/  (nothing else is written)
"""

import argparse
import hashlib
import json
import os
import shutil

import tensorflow as tf
import yaml

W = "/workspace"
OUT = W + "/runs/variants_verify/l1/fixtures"
SRC = {
    "textures_db": W + "/runs/deeptex/datasets/textures_db/textures_db_lrx4.tfrecord",
    "grouped": W + "/runs/deeptex/datasets/matsynth_grouped/matsynth_grouped_lrx4.tfrecord",
}
VAL = W + "/runs/deeptex/datasets/textures_db"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cut(name, n):
    src = SRC[name]
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.basename(src))
    with tf.io.TFRecordWriter(dst) as w:
        for i, rec in enumerate(tf.data.TFRecordDataset(src).take(n)):
            w.write(rec.numpy())
    with open(src + ".meta.json") as f:
        meta = json.load(f)
    meta.update(tfrecord=dst, record_count=n, bytes=os.path.getsize(dst), sha256=sha256(dst),
                l1_fixture_of=src)
    with open(dst + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(d, "verify_report.json"), "w") as f:
        json.dump({"verdict": "PASS", "l1_fixture": True,
                   "meta": {k: meta[k] for k in ("sha256", "bytes", "record_count")}}, f, indent=2)
    return dst


def val_tiles(k):
    for sub in ("HR_val", "LRx4_val"):
        os.makedirs(os.path.join(OUT, "val", sub), exist_ok=True)
    names = sorted(os.listdir(os.path.join(VAL, "HR_val")))[:k]
    for n in names:
        for sub in ("HR_val", "LRx4_val"):
            shutil.copy(os.path.join(VAL, sub, n), os.path.join(OUT, "val", sub, n))


def config(src_rel, dst_name, record, batch, nb, **extra):
    with open(os.path.join(W, src_rel)) as f:
        cfg = yaml.safe_load(f) if False else yaml.load(f, Loader=yaml.Loader)
    cfg["batch_size"] = batch
    cfg["network_G"]["nb"] = nb
    cfg["train_dataset"]["path"] = record
    cfg.pop("val_dataset", None)
    cfg.update(extra)
    os.makedirs(os.path.join(OUT, "configs"), exist_ok=True)
    p = os.path.join(OUT, "configs", dst_name)
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=12)
    ap.add_argument("--nb", type=int, default=2, help="RRDB blocks in the CPU configs")
    args = ap.parse_args()
    tdb = cut("textures_db", args.records)
    grp = cut("grouped", max(4, args.records // 3))
    val_tiles(3)
    config("configs/deeptex/psnr_texsr.yaml", "psnr.yaml", tdb, 2, args.nb)
    config("configs/deeptex/esrgan_scaled.yaml", "esrgan.yaml", tdb, 2, args.nb)
    config("configs/deeptex/esrgan_asconfigured.yaml", "asconfigured.yaml", tdb, 2, args.nb)
    config("external/DeepTextures/configs/esrgan.yaml", "paper.yaml", tdb, 2, args.nb,
           niter=5, save_steps=5)
    for arm in ("r0a", "r1"):
        config("configs/deeptex/esrgan_matsynth_render_%s.yaml" % arm, "render_%s.yaml" % arm,
               grp, 1, args.nb,
               checkpoint_root="/workspace/runs/variants_verify/l1/_cfgroot/checkpoints",
               log_root="/workspace/runs/variants_verify/l1/_cfgroot/logs")
    print("[l1-fixtures] %s" % OUT)


if __name__ == "__main__":
    main()
