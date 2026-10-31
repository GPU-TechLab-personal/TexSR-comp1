#!/usr/bin/env python3
"""The MatSynth arms' training mixes, and the one TFRecord each is packed into.

Runs in the deeptex container:

    python /workspace/variants/deeptex/matsynth/recipes.py --recipe chratio \\
        --out /workspace/runs/deeptex/datasets/matsynth/matsynth_chratio_lrx4.tfrecord \\
        --pairs-manifest /workspace/runs/deeptex/datasets/matsynth/pairs_manifest.json --overwrite

A recipe is how many times each map's pair folder (matsynth_make_pairs.py --out
<root>/<map>/{HR,LRx4}) is packed into the record:

  all        every map once                     (matsynth_all, esrgan_matsynth.yaml)
  chratio    baseColor and normal three times   (matsynth_chratio): a map's share
             each - its share proportional to   proportional to its real channel
             the channel count, 3:3:1:1:1       count
  basecolor  baseColor only                     (matsynth_basecolor)

The table is the old run_deeptex.ps1:539-543 `$script:MatsynthRecipes`, the
mechanism the old texsr_build_tfrecord.py `--repeat` (:37-43, :86-88, :102-105,
:116, :133-135): copies after the first get "~r2".."~rk" appended to img_name,
because texsr_verify_tfrecord.py fails on a duplicate name, and every copy goes
through the same single shuffle as everything else. The ps1 passed no --repeat
for "all"; with every k = 1 the list and its permutation are the same, and the
meta's "repeat" was [1]*5 either way - so the "all" record is byte-for-byte the
one the old chain wrote.

The packing itself is upstream_runner/texsr_build_tfrecord.build(), unchanged.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.deeptex.upstream_runner import texsr_build_tfrecord as builder  # noqa: E402

# Map folder -> times packed. Insertion order is the --hr/--lr order the ps1 passed,
# and that order is part of the shuffle's input, so it is part of the recipe.
RECIPES = {
    "all":       {"baseColor": 1, "normal": 1, "roughness": 1, "metallic": 1, "height": 1},
    "chratio":   {"baseColor": 3, "normal": 3, "roughness": 1, "metallic": 1, "height": 1},
    "basecolor": {"baseColor": 1},
}
DEFAULT_ROOT = "/workspace/runs/deeptex/datasets/matsynth"


def sources(recipe, root):
    """-> (hr dirs, lr dirs, repeats), in the recipe's order."""
    mix = RECIPES[recipe]
    hr = [os.path.join(root, m, "HR") for m in mix]
    lr = [os.path.join(root, m, "LRx4") for m in mix]
    return hr, lr, [mix[m] for m in mix]


def main():
    ap = builder.parser(__doc__)
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="matsynth_make_pairs.py --out: <root>/<map>/{HR,LRx4}")
    args = ap.parse_args()

    args.hr, args.lr, repeats = sources(args.recipe, args.root)
    if len(repeats) != len(args.hr) or min(repeats) < 1:
        raise SystemExit("--repeat needs one value >= 1 per --hr/--lr pair, got %s for %d pair(s)"
                         % (repeats, len(args.hr)))
    print("  recipe %s : %s" % (args.recipe, ", ".join(
        "%s x%d" % (m, k) for m, k in RECIPES[args.recipe].items())))
    builder.build(args, repeats)


if __name__ == "__main__":
    main()
