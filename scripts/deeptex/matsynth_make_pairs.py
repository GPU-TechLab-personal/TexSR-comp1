#!/usr/bin/env python3
"""Build DeepTextures training pairs from MatSynth's TRAIN split.

Run it in the dataprep container (pyarrow + cv2, no TensorFlow):

    docker compose run --rm dataprep python scripts/deeptex/matsynth_make_pairs.py --check-eval
    docker compose run --rm dataprep python scripts/deeptex/matsynth_make_pairs.py --jobs 4

Layout produced under --out:

    <map>/HR/<material>__<map>.png      512^2, 8-bit, 3-channel
    <map>/LRx4/<material>__<map>.png    128^2, 8-bit, 3-channel
    _meta/<material>.json               per-material record (makes reruns resume)
    pairs_manifest.json                 what was kept, excluded and skipped, and why

<map> is baseColor, normal, roughness, metallic, height - the same keys
deeptex_compare.py uses on the evaluation side. File names carry the map so that
texsr_build_tfrecord.py can pack all five folders into ONE record without name
clashes: the paper trains a single network on all five channel types
(docs/DEEPTEX-DEVIATIONS.md 0.3), and that is the arm this feeds.

WHY THESE PAIRS MATCH THE EVALUATION
------------------------------------
The 12 evaluation materials under runs/matsynth were built by
scripts/common/matsynth_extract.py --mode resize: each source map (4096^2 or
2048^2) is area-decimated to 512^2 for GT and, separately, to 128^2 for LR. The
pairs here are built by calling THAT SCRIPT'S functions in the same order -
degrade() below is extract_row()'s resize branch - and then encoded with
map_encode.to_rgb8, which deeptex_compare.load_as_rgb01 also calls. So the network
trains on exactly the degradation and the byte encoding it is evaluated on.
--check-eval proves it rather than asserting it: it rebuilds the 12 from the test
shard and compares every pixel against runs/matsynth.

This also retires a mismatch the textures_db arm has: there, LR is bicubic
(texsr_make_pairs.py) while the MatSynth eval LR is area.

WHAT IS EXCLUDED
----------------
* The whole test split - only train-*.parquet is read. All 12 evaluation
  materials are in test-00000.
* MatSynth's own blends (ms_*) built from an evaluation material. Their names are
  "ms_<partA>__<partB>" with the source prefix (acg_, th_, ...) stripped, and a
  part may carry a variant suffix (concrete_042_d), so a part matches when it
  equals the evaluation material's stem or starts with stem + "_". Measured on
  the local shards: exactly one hit, ms_concrete_026__moss_001 (acg_concrete_026).
* Not excluded, only logged: other sources whose names share a stem by
  coincidence of numbering (tc_concrete_026 is TextureCan, the evaluation
  material is ambientCG's acg_concrete_026).

Shards that are still git-lfs pointer stubs (~134 bytes) are skipped and counted.

KNOWN LIMITS, carried into the arm on purpose (DEEPTEX-DEVIATIONS 0.9)
---------------------------------------------------------------------
* height is 16-bit in MatSynth and becomes 8-bit here: the upstream loader
  decodes PNGs as uint8 (modules/dataset.py:12-13).
* Grey maps are replicated into three channels, which is how the paper's
  3-channel network sees them.
* Constant maps (most metallic maps are all zero) are kept - the paper trained on
  "textures x 5 channels" with no curation - and counted in the manifest.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import time

import cv2
import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "common"))
import map_encode  # noqa: E402
from matsynth_extract import (MAPS, as_integer, decode, fit_channels,  # noqa: E402
                              resize_square)

LFS_STUB_BYTES = 1 << 20
SOURCE_PREFIX = re.compile(r"^(acg|cgbc|js|st|tc|th|\d+)_")
COLUMNS = ["name", "category"] + list(MAPS)


def map_key(column):
    """MatSynth column -> the map name used everywhere downstream (baseColor, ...)."""
    return os.path.splitext(MAPS[column][0])[0]


def safe_name(name):
    # Same rule as matsynth_extract.extract_row, so a material has one name
    # on both the training and the evaluation side.
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def degrade(src, channels, hr_size, lr_size):
    """matsynth_extract.extract_row, --mode resize branch, minus the file I/O.

    Kept call-for-call identical on purpose. --check-eval compares the result
    against the files that function wrote.
    """
    arr = fit_channels(src, channels)
    return (as_integer(resize_square(arr, hr_size)),
            as_integer(resize_square(arr, lr_size)))


# ------------------------------------------------------------------ exclusions

def stem(name):
    return SOURCE_PREFIX.sub("", name, count=1)


def load_eval_names(eval_root):
    names = []
    for meta in sorted(glob.glob(os.path.join(eval_root, "*", "meta.json"))):
        with open(meta) as f:
            m = json.load(f)
        if m.get("mode", "resize") != "resize":
            raise SystemExit("%s was extracted with mode=%s; the pairs here assume resize"
                             % (meta, m.get("mode")))
        names.append(m["name"])
    if not names:
        raise SystemExit("no evaluation materials under %s" % eval_root)
    return names


def exclusion_reason(name, eval_names, eval_stems):
    if name in eval_names:
        return "is an evaluation material"
    if name.startswith("ms_"):
        for part in name[3:].split("__"):
            for s, ev in eval_stems.items():
                if part == s or part.startswith(s + "_"):
                    return "MatSynth blend of evaluation material %s" % ev
    return None


def near_name(name, eval_stems):
    if name.startswith("ms_"):
        return None
    return eval_stems.get(stem(name))


# ------------------------------------------------------------------ writing

def write_rgb_png(path, rgb):
    # cv2 writes BGR; the network reads RGB through tf.image.decode_png, so the
    # reversal makes the FILE hold RGB in the right order. Atomic so a killed run
    # never leaves a truncated PNG that looks finished.
    tmp = path[:-4] + ".partial.png"
    if not cv2.imwrite(tmp, np.ascontiguousarray(rgb[:, :, ::-1]),
                       [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError("cv2.imwrite failed for " + tmp)
    os.replace(tmp, path)


def out_paths(out, safe, key):
    fname = "%s__%s.png" % (safe, key)
    return (os.path.join(out, key, "HR", fname), os.path.join(out, key, "LRx4", fname))


def material_done(out, safe):
    meta = os.path.join(out, "_meta", safe + ".json")
    if not os.path.exists(meta):
        return None
    with open(meta) as f:
        rec = json.load(f)
    for key in rec.get("maps", {}):
        if not all(os.path.exists(p) for p in out_paths(out, safe, key)):
            return None
    return rec


def process_shard(job):
    """One shard -> pairs on disk + one record per material.

    Reads ONE COLUMN AT A TIME. Each shard is a single ~1 GB row group, and
    iter_batches over all five map columns materialises that whole group per
    worker: six workers did exactly that and died with ENOMEM inside a 16 GB
    Docker VM. One column of one shard is a few hundred MB, and only the small
    512^2/128^2 results are held across columns. That alone was not enough -
    Arrow's allocator keeps what a worker has read, so main() also gives every
    shard a fresh process (the same two fixes experiments/common/shards.py
    arrived at on 2026-09-29).
    """
    path, cfg = job
    shard = os.path.basename(path)
    eval_names = set(cfg["eval_names"])
    eval_stems = cfg["eval_stems"]
    out = cfg["out"]
    pf = pq.ParquetFile(path, memory_map=True)
    head = pf.read(columns=["name", "category"])
    names = head.column("name").to_pylist()
    cats = head.column("category").to_pylist()

    records, todo = {}, []
    for position, (name, cat) in enumerate(zip(names, cats)):
        safe = safe_name(name)
        base = {"name": name, "safe": safe, "shard": shard, "row": position, "category": cat}
        reason = exclusion_reason(name, eval_names, eval_stems)
        if reason:
            records[position] = dict(base, status="excluded", reason=reason)
            continue
        done = None if cfg["force"] else material_done(out, safe)
        if done:
            records[position] = dict(done, status="kept", resumed=True)
            continue
        records[position] = base
        todo.append(position)

    maps = {i: {} for i in todo}
    problems = {}
    for column, (_, channels) in MAPS.items():
        if not todo:
            break
        col = pf.read(columns=[column]).column(column)
        for i in todo:
            if i in problems:
                continue
            payload = col[i].as_py()
            if payload is None or payload.get("bytes") is None:
                problems[i] = "no %s map" % column
                continue
            src = decode(payload["bytes"])
            h, w = src.shape[:2]
            if h != w:
                problems[i] = "%s is not square (%dx%d)" % (column, w, h)
                continue
            if h < cfg["hr_size"]:
                problems[i] = "%s is %d^2, smaller than HR %d^2" % (column, h, cfg["hr_size"])
                continue
            hr, lr = degrade(src, channels, cfg["hr_size"], cfg["lr_size"])
            hr8, lr8 = map_encode.to_rgb8(hr), map_encode.to_rgb8(lr)
            maps[i][map_key(column)] = (hr8, lr8, {
                "source": "%dx%d %s ch=%d" % (w, h, src.dtype, 1 if src.ndim == 2 else src.shape[2]),
                "constant": bool(hr8.max() == hr8.min()),
                "hr_min": int(hr8.min()), "hr_max": int(hr8.max()),
            })
            del src, hr, lr
        del col

    for i in todo:
        base = records[i]
        if i in problems:
            records[i] = dict(base, status="skipped", reason=problems[i])
            continue
        rec = dict(base, maps={k: v[2] for k, v in maps[i].items()})
        near = near_name(base["name"], eval_stems)
        if near:
            rec["near_eval_name"] = near
        for key, (hr8, lr8, _) in maps[i].items():
            hr_p, lr_p = out_paths(out, base["safe"], key)
            write_rgb_png(hr_p, hr8)
            write_rgb_png(lr_p, lr8)
        # The sidecar goes last: its presence is what marks the material complete.
        with open(os.path.join(out, "_meta", base["safe"] + ".json"), "w") as f:
            json.dump(rec, f, indent=1)
        records[i] = dict(rec, status="kept")
    return shard, [records[i] for i in sorted(records)]


# ------------------------------------------------------------------ check-eval

def check_eval(args, eval_names):
    """Rebuild the evaluation materials from their test shard and diff every pixel.

    If this fails, the network would be trained on a degradation it is not
    evaluated on. Do not build the training set until it passes.
    """
    pf = pq.ParquetFile(args.check_shard)
    want = set(eval_names)
    seen, bad = 0, []
    for position, row in enumerate(pf.iter_batches(batch_size=1, columns=COLUMNS)):
        name = row.column("name")[0].as_py()
        if name not in want:
            continue
        safe = safe_name(name)
        for column, (filename, channels) in MAPS.items():
            src = decode(row.column(column)[0].as_py()["bytes"])
            hr, lr = degrade(src, channels, args.hr_size, args.lr_size)
            for got, sub in ((hr, "GT"), (lr, "SVBRDF")):
                ref = cv2.imread(os.path.join(args.eval_root, safe, sub, filename),
                                 cv2.IMREAD_UNCHANGED)
                same = (ref is not None and ref.dtype == got.dtype
                        and ref.shape == got.shape and np.array_equal(ref, got))
                # the encoded bytes are what the network actually sees
                same = same and np.array_equal(map_encode.to_rgb8(ref), map_encode.to_rgb8(got))
                if not same:
                    bad.append("%s/%s/%s" % (safe, sub, filename))
        seen += 1
        print("  [%2d] %-28s checked" % (position, name))
    print("checked %d of %d evaluation materials, %d map(s) x 2 sizes each"
          % (seen, len(want), len(MAPS)))
    if seen != len(want):
        print("[FAIL] %d evaluation material(s) not found in %s"
              % (len(want) - seen, os.path.basename(args.check_shard)))
        return 1
    if bad:
        for b in bad[:20]:
            print("[FAIL] differs: %s" % b)
        print("[FAIL] %d file(s) differ - training pairs would not match the eval degradation"
              % len(bad))
        return 1
    print("[PASS] training-pair degradation is pixel-identical to runs/matsynth "
          "on all %d files" % (seen * len(MAPS) * 2))
    return 0


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/MatSynth/data")
    ap.add_argument("--out", default="/workspace/runs/deeptex/datasets/matsynth")
    ap.add_argument("--eval-root", default="/workspace/runs/matsynth",
                    help="the evaluation materials; excluded from training")
    ap.add_argument("--hr-size", type=int, default=512)
    ap.add_argument("--lr-size", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=4,
                    help="shards processed in parallel; each holds one map column "
                         "of one shard at a time (a few hundred MB)")
    ap.add_argument("--limit-shards", type=int, default=0, help="first N real shards only")
    ap.add_argument("--timeout", type=int, default=900,
                    help="seconds to wait for any shard before declaring a worker lost")
    ap.add_argument("--force", action="store_true", help="rebuild materials already done")
    ap.add_argument("--check-eval", action="store_true",
                    help="verify the degradation against --eval-root and exit")
    ap.add_argument("--check-shard", default="/data/MatSynth/data/test-00000-of-00008.parquet")
    args = ap.parse_args()

    eval_names = load_eval_names(args.eval_root)
    if args.check_eval:
        sys.exit(check_eval(args, eval_names))

    shards = sorted(glob.glob(os.path.join(args.data, "train-*.parquet")))
    if not shards:
        raise SystemExit("no train shards under %s" % args.data)
    real = [s for s in shards if os.path.getsize(s) >= LFS_STUB_BYTES]
    stubs = len(shards) - len(real)
    if args.limit_shards:
        real = real[:args.limit_shards]
    print("%d train shard(s): %d real, %d git-lfs stub(s) skipped; processing %d"
          % (len(shards), len(shards) - stubs, stubs, len(real)))

    for key in (map_key(c) for c in MAPS):
        for sub in ("HR", "LRx4"):
            os.makedirs(os.path.join(args.out, key, sub), exist_ok=True)
    os.makedirs(os.path.join(args.out, "_meta"), exist_ok=True)
    # A killed run leaves *.partial.png behind; texsr_build_tfrecord.py packs every
    # *.png it finds, so they must not survive into the next build.
    stale = glob.glob(os.path.join(args.out, "*", "*", "*.partial.png"))
    for p in stale:
        os.remove(p)
    if stale:
        print("removed %d stale .partial.png file(s) from an interrupted run" % len(stale))

    cfg = {"out": args.out, "hr_size": args.hr_size, "lr_size": args.lr_size,
           "force": args.force, "eval_names": eval_names,
           "eval_stems": {stem(n): n for n in eval_names}}
    t0 = time.time()
    records = []
    # maxtasksperchild=1: one process per shard, so Arrow's retained buffers die
    # with it. The timeout: Pool never returns the task of a worker the kernel
    # OOM-killed, so without it the run would hang at 0% CPU instead of failing.
    with mp.Pool(args.jobs, maxtasksperchild=1) as pool:
        it = pool.imap_unordered(process_shard, [(s, cfg) for s in real])
        for i in range(1, len(real) + 1):
            try:
                shard, recs = it.next(timeout=args.timeout)
            except mp.TimeoutError:
                raise SystemExit("no shard finished for %d s - a worker was most likely "
                                 "OOM-killed. Rerun the same command (finished materials "
                                 "are skipped), with --jobs %d." % (args.timeout, max(1, args.jobs - 1)))
            records.extend(recs)
            kept = sum(r["status"] == "kept" for r in recs)
            print("  [%3d/%d] %s  kept %d/%d   (%.0f s)"
                  % (i, len(real), shard, kept, len(recs), time.time() - t0), flush=True)

    records.sort(key=lambda r: (r["shard"], r["row"]))
    kept = [r for r in records if r["status"] == "kept"]
    excluded = [r for r in records if r["status"] == "excluded"]
    skipped = [r for r in records if r["status"] == "skipped"]

    # Leakage, asserted on the result rather than trusted from the reader.
    names = [r["name"] for r in kept]
    assert len(names) == len(set(names)), "duplicate material names in the kept set"
    assert not set(names) & set(eval_names), "an evaluation material was kept"
    assert all(r["shard"].startswith("train-") for r in kept), "a non-train shard was read"

    keys = [map_key(c) for c in MAPS]
    constant = {k: sum(r["maps"][k]["constant"] for r in kept) for k in keys}
    counts = {}
    for k in keys:
        n_hr = len(glob.glob(os.path.join(args.out, k, "HR", "*__%s.png" % k)))
        n_lr = len(glob.glob(os.path.join(args.out, k, "LRx4", "*__%s.png" % k)))
        counts[k] = {"HR": n_hr, "LRx4": n_lr}
        if n_hr != len(kept) or n_lr != len(kept):
            raise SystemExit("%s: %d HR / %d LR files on disk but %d materials kept - "
                             "stale files from another run? rerun with --force into a clean --out"
                             % (k, n_hr, n_lr, len(kept)))

    manifest = {
        "recipe": "matsynth_extract.py --mode resize degradation (INTER_AREA from source "
                  "to %d and to %d), then map_encode.to_rgb8" % (args.hr_size, args.lr_size),
        "hr_size": args.hr_size, "lr_size": args.lr_size,
        "data": args.data,
        "shards": {"train_total": len(shards), "real": len(shards) - stubs, "stubs": stubs,
                   "processed": [os.path.basename(s) for s in real]},
        "eval_materials": eval_names,
        "materials": {"seen": len(records), "kept": len(kept),
                      "excluded": len(excluded), "skipped": len(skipped)},
        "files_per_map": counts,
        "constant_maps": constant,
        "excluded": [{"name": r["name"], "reason": r["reason"]} for r in excluded],
        "skipped": [{"name": r["name"], "shard": r["shard"], "reason": r["reason"]}
                    for r in skipped],
        "near_eval_names_kept": [{"name": r["name"], "eval": r["near_eval_name"]}
                                 for r in kept if r.get("near_eval_name")],
        "kept": [{k: r[k] for k in ("name", "shard", "row", "category")} for r in kept],
    }
    with open(os.path.join(args.out, "pairs_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print()
    print("kept %d material(s) x %d maps = %d pairs   (%.0f s)"
          % (len(kept), len(keys), len(kept) * len(keys), time.time() - t0))
    for r in excluded:
        print("  [excluded] %-36s %s" % (r["name"], r["reason"]))
    for r in manifest["near_eval_names_kept"]:
        print("  [kept, near-name] %-28s ~ %s (different source)" % (r["name"], r["eval"]))
    for r in skipped[:20]:
        print("  [skipped] %-36s %s" % (r["name"], r["reason"]))
    print("  constant maps: " + ", ".join("%s %d" % kv for kv in constant.items()))
    print("manifest -> %s" % os.path.join(args.out, "pairs_manifest.json"))


if __name__ == "__main__":
    main()
