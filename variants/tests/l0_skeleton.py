#!/usr/bin/env python3
"""L0 skeleton: loop.py is train_esrgan.py plus hook calls, and nothing else.

For each (upstream, ours) pair below, diff the two files line by line with
whitespace normalised and blank lines dropped (a hook's `with` block re-indents
the two apply_gradients, which is not a change). Then:

  * every line ours ADDS must call into the hooks (contain "hooks");
  * every line ours REMOVES must occur verbatim in paper_loop/hooks.py, where the
    default implementation (or the comment beside it) keeps it.

So deleting the hook calls from loop.py and substituting each hook's default
gives back train_esrgan.py. --write regenerates the committed unified diffs
(paper_loop/upstream.diff, psnr_upstream.diff) and fails if they were stale.
"""

import argparse
import difflib
import os
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir))
LOOP = os.path.join(REPO, "variants", "deeptex", "paper_loop")
PAIRS = [
    ("external/DeepTextures/train_esrgan.py", "variants/deeptex/paper_loop/loop.py",
     "variants/deeptex/paper_loop/upstream.diff"),
    ("external/esrgan-tf2/train_psnr.py", "variants/deeptex/paper_loop/psnr_loop.py",
     "variants/deeptex/paper_loop/psnr_upstream.diff"),
]
HOOKS = "variants/deeptex/paper_loop/hooks.py"


def lines(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read().splitlines()


def norm(ls):
    return [" ".join(l.split()) for l in ls if l.strip()]


def check(up, ours, hooks_src):
    a, b = norm(lines(up)), norm(lines(ours))
    problems, added, removed = [], 0, 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        for ln in b[j1:j2]:
            added += 1
            if "hooks" not in ln:
                problems.append("added line is not a hook call: %r" % ln)
        for ln in a[i1:i2]:
            removed += 1
            if ln not in hooks_src:
                problems.append("removed upstream line not kept in hooks.py: %r" % ln)
    return problems, added, removed


def unified(up, ours):
    return "".join(difflib.unified_diff(
        [l + "\n" for l in lines(up)], [l + "\n" for l in lines(ours)],
        fromfile="a/" + up, tofile="b/" + ours, n=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="regenerate the committed .diff files")
    args = ap.parse_args()
    hooks_src = " ".join(" ".join(l.split()) for l in lines(HOOKS))
    n_fail = 0
    for up, ours, diff_rel in PAIRS:
        probs, added, removed = check(up, ours, hooks_src)
        text = unified(up, ours)
        diff_path = os.path.join(REPO, diff_rel)
        old = open(diff_path, encoding="utf-8").read() if os.path.isfile(diff_path) else None
        if args.write:
            with open(diff_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
        elif old != text:
            probs.append("%s is stale - rerun with --write" % diff_rel)
        v = "PASS" if not probs else "FAIL"
        n_fail += bool(probs)
        print("%s %s vs %s: %d lines added (all hook calls), %d upstream lines moved into hooks.py"
              % (v, ours, up, added, removed))
        for p in probs:
            print("       - %s" % p)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
