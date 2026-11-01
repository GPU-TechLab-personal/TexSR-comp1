#!/usr/bin/env python3
"""Gate data: runs/sr_ft/pairs is what both trainers will read, and it is clean.

Runs in the dataprep service (it reuses matsynth_make_pairs.py's exclusion code,
which imports pyarrow):

    docker compose run --rm dataprep python variants/sr_finetune/gates/gate_data.py

-> runs/sr_ft/gate/data.json ("pass" is what Show-GateResults reads).

  counts    HR and LRx4 hold 14,520 files each, 2,904 per map
  pairing   the sorted name lists of HR and LRx4 are identical - KAIR pairs by
            sorted position, BasicSR by name, so both pair the same files
  format    every PNG header: HR 512x512, LRx4 128x128, 8-bit RGB (colour type 2) -
            what both trainers' 3-channel loaders expect
  bytes     every file has its source's size; a seeded sample of --sample files per
            side has its source's sha256 (prep_pairs.py hard-links, so this is the
            check that nothing replaced a link since)
  eval      no material is an evaluation material or a MatSynth blend of one, by
            matsynth_make_pairs.py's own load_eval_names / exclusion_reason over the
            materials' real names (_meta/<safe>.json)
"""
import argparse
import hashlib
import json
import os
import random
import struct
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.util import paths  # noqa: E402
from variants.deeptex.matsynth import matsynth_make_pairs as mk  # noqa: E402

MAPS = ("baseColor", "normal", "roughness", "metallic", "height")
SRC = os.path.join(paths.REPO, "runs", "deeptex", "datasets", "matsynth")
DST = os.path.join(paths.REPO, "runs", "sr_ft", "pairs")
GATE_DIR = os.path.join(paths.REPO, "runs", "sr_ft", "gate")
SIZE = {"HR": 512, "LRx4": 128}


def png_header(path):
    with open(path, "rb") as f:
        b = f.read(26)
    w, h = struct.unpack(">II", b[16:24])
    return w, h, b[24], b[25]          # width, height, bit depth, colour type


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def source_of(name):
    return os.path.join(SRC, name.rsplit("__", 1)[1][:-4], "{side}", name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    names = {s: sorted(f for f in os.listdir(os.path.join(DST, s)) if f.endswith(".png")) for s in SIZE}
    checks = {}
    per_map = {m: sum(n.endswith("__%s.png" % m) for n in names["HR"]) for m in MAPS}
    checks["counts"] = {"ok": len(names["HR"]) == len(names["LRx4"]) == 14520 and all(v == 2904 for v in per_map.values()),
                        "HR": len(names["HR"]), "LRx4": len(names["LRx4"]), "per_map": per_map}
    checks["pairing"] = {"ok": names["HR"] == names["LRx4"],
                         "first_mismatch": next(([a, b] for a, b in zip(names["HR"], names["LRx4"]) if a != b), None)}

    bad_fmt, bad_size = [], []
    for s, px in SIZE.items():
        for n in names[s]:
            p = os.path.join(DST, s, n)
            if png_header(p) != (px, px, 8, 2):
                bad_fmt.append("%s/%s %s" % (s, n, png_header(p)))
            if os.path.getsize(p) != os.path.getsize(source_of(n).format(side=s)):
                bad_size.append("%s/%s" % (s, n))
    checks["format"] = {"ok": not bad_fmt, "bad": bad_fmt[:10], "n_bad": len(bad_fmt)}
    rng = random.Random(args.seed)
    bad_sha = []
    for s in SIZE:
        for n in rng.sample(names[s], min(args.sample, len(names[s]))):
            if sha256(os.path.join(DST, s, n)) != sha256(source_of(n).format(side=s)):
                bad_sha.append("%s/%s" % (s, n))
    checks["bytes"] = {"ok": not bad_size and not bad_sha, "size_mismatch": bad_size[:10],
                       "sha_mismatch": bad_sha[:10], "sha_sampled": 2 * min(args.sample, len(names["HR"]))}

    eval_names = mk.load_eval_names(os.path.join(paths.REPO, "runs", "matsynth"))
    eval_stems = {mk.stem(n): n for n in eval_names}
    safes = sorted({n.rsplit("__", 1)[0] for n in names["HR"]})
    hits, unnamed = [], []
    for safe in safes:
        meta = os.path.join(SRC, "_meta", safe + ".json")
        if not os.path.exists(meta):
            unnamed.append(safe)
            continue
        with open(meta) as f:
            real = json.load(f)["name"]
        why = mk.exclusion_reason(real, eval_names, eval_stems)
        if why:
            hits.append("%s: %s" % (real, why))
    checks["eval"] = {"ok": not hits and not unnamed, "materials": len(safes), "eval_materials": eval_names,
                      "leaks": hits, "no_meta": unnamed[:10]}

    report = {"pairs": DST.replace(paths.REPO, "").lstrip("/\\"), "checks": checks,
              "failed": [k for k, c in checks.items() if not c["ok"]]}
    report["pass"] = not report["failed"]
    os.makedirs(GATE_DIR, exist_ok=True)
    path = os.path.join(GATE_DIR, "data.json")
    with open(path + ".partial", "w") as f:
        json.dump(report, f, indent=2)
    os.replace(path + ".partial", path)
    for k, c in checks.items():
        print("  %-8s %s" % (k, "ok" if c["ok"] else "FAIL"))
    print("[gate data] %s -> %s" % ("PASS" if report["pass"] else "FAIL", path))
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
