#!/usr/bin/env python3
"""Exact comparison of the latest checkpoint in two directories (deeptex container).

    python l1_ckpt_compare.py DIR_A DIR_B   -> exit 0 if every variable is equal

Compares variable NAMES and VALUES (np.array_equal, bit-exact for floats via
tobytes), ignoring the object-graph proto itself.
"""

import sys

import numpy as np
import tensorflow as tf


def load(d):
    path = tf.train.latest_checkpoint(d)
    if path is None:
        sys.exit("no checkpoint in %s" % d)
    r = tf.train.load_checkpoint(path)
    names = sorted(n for n in r.get_variable_to_shape_map() if n != "_CHECKPOINTABLE_OBJECT_GRAPH")
    return path, {n: r.get_tensor(n) for n in names}


def main():
    pa, a = load(sys.argv[1])
    pb, b = load(sys.argv[2])
    bad, maxabs = [], 0.0
    if set(a) != set(b):
        bad.append("variable sets differ: only A %s, only B %s"
                   % (sorted(set(a) - set(b))[:5], sorted(set(b) - set(a))[:5]))
    for n in sorted(set(a) & set(b)):
        x, y = np.asarray(a[n]), np.asarray(b[n])
        if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
            diff = (float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                    if x.shape == y.shape and x.dtype.kind in "fiu" else float("nan"))
            bad.append("%s differs (max |d| = %g)" % (n, diff))
            if diff == diff:
                maxabs = max(maxabs, diff)
    print("[ckpt] %s vs %s: %d variables, %d differ, maxabs=%g" % (pa, pb, len(a), len(bad), maxabs))
    for x in bad[:10]:
        print("       %s" % x)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
