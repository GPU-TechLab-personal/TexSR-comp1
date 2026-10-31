#!/usr/bin/env python3
"""Pixel difference between two deeptex_compare --out trees (dataprep container).

    python variants/tests/l3_pngdiff.py REF NEW --json out.json [--material m ...]

For every <material>/<map>.png in NEW: the largest absolute difference against the
same file in REF in 8-bit levels (16-bit files are compared in their own units
and reported as such), the number of differing pixels, and whether the two
sr_manifest.json agree once "checkpoint" is left out (a GPU rerun of the same
checkpoint; the step must match).
"""

import argparse
import json
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    rows, worst, n_files, n_bytes_same = [], 0, 0, 0
    problems = []
    for mat in sorted(os.listdir(args.new)):
        nd = os.path.join(args.new, mat)
        if not os.path.isdir(nd):
            continue
        rd = os.path.join(args.ref, mat)
        for f in sorted(os.listdir(nd)):
            if not f.endswith(".png"):
                continue
            n_files += 1
            a = cv2.imread(os.path.join(rd, f), cv2.IMREAD_UNCHANGED)
            b = cv2.imread(os.path.join(nd, f), cv2.IMREAD_UNCHANGED)
            if a is None or b is None or a.shape != b.shape or a.dtype != b.dtype:
                problems.append("%s/%s: missing or shape/dtype differs" % (mat, f))
                continue
            d = np.abs(a.astype(np.int64) - b.astype(np.int64))
            mx = int(d.max())
            worst = max(worst, mx)
            n_bytes_same += int(mx == 0)
            rows.append({"file": "%s/%s" % (mat, f), "max_abs": mx,
                         "n_diff": int((d.reshape(d.shape[0], d.shape[1], -1).max(axis=2) > 0).sum()),
                         "pixels": int(d.shape[0] * d.shape[1])})
        ma, mb = os.path.join(rd, "sr_manifest.json"), os.path.join(nd, "sr_manifest.json")
        if os.path.exists(ma) and os.path.exists(mb):
            ja, jb = json.load(open(ma)), json.load(open(mb))
            for j in (ja, jb):
                j.pop("checkpoint", None)
                for o in j.get("outputs", {}).values():
                    o.pop("channel_spread", None)      # raw float output: moves with the GPU
            if ja != jb:
                problems.append("%s/sr_manifest.json differs (beyond checkpoint/channel_spread)" % mat)
    res = {"ref": args.ref, "new": args.new, "files": n_files, "pixel_identical_files": n_bytes_same,
           "max_abs_diff": worst, "problems": problems,
           "worst": sorted(rows, key=lambda r: -r["max_abs"])[:10]}
    with open(args.json, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("files", "pixel_identical_files", "max_abs_diff", "problems")}))


if __name__ == "__main__":
    main()
