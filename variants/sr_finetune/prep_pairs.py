#!/usr/bin/env python3
"""The fine-tuning pairs: the five MatSynth maps' existing pairs in ONE folder pair.

Host, stdlib only (hard links are made where the files live, on NTFS):

    python variants/sr_finetune/prep_pairs.py

    runs/deeptex/datasets/matsynth/<map>/{HR,LRx4}/<material>__<map>.png   (matsynth_make_pairs.py)
 -> runs/sr_ft/pairs/{HR,LRx4}/<material>__<map>.png                       14,520 pairs

The same files DeepTextures' matsynth_all arm trained on - not rebuilt, linked:
the area degradation and map_encode.to_rgb8 encoding the evaluation uses, grey
maps replicated to three channels, height 16 -> 8 bit, the evaluation materials
and their MatSynth blends already excluded (matsynth_make_pairs.py, its
pairs_manifest.json). One folder for all five maps, so a batch mixes them 1:1 as
matsynth_all does; the names already carry the map, so nothing collides.

Both trainers read a flat folder pair: BasicSR pairs LR with HR by NAME
(basicsr/data/data_util.py:198-231), KAIR by SORTED POSITION
(data/dataset_sr.py:28-33, utils/utils_image.py:67-82). The names in HR and LRx4
are identical, so both pair the same files.

A hard link costs no space and is byte-identical by construction; where linking
fails (another volume, a filesystem without links) the file is copied and its
sha256 compared. Re-running is a no-op for files already linked to their source;
a destination file that is NOT its source stops the run (exit 2) unless --force.
-> runs/sr_ft/pairs/manifest.json; gates/gate_data.py checks the result.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 2)))
from variants.util import paths  # noqa: E402

MAPS = ("baseColor", "normal", "roughness", "metallic", "height")
SIDES = ("HR", "LRx4")
SRC = os.path.join(paths.REPO, "runs", "deeptex", "datasets", "matsynth")
DST = os.path.join(paths.REPO, "runs", "sr_ft", "pairs")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def place(src, dst, force):
    """'linked' | 'already' | 'copied'."""
    if os.path.exists(dst):
        if os.path.samefile(src, dst):
            return "already"
        if not force:
            raise SystemExit("[fatal] %s exists and is not a link to %s (--force replaces it)" % (dst, src))
        os.remove(dst)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copyfile(src, dst)
        if sha256(src) != sha256(dst):
            raise SystemExit("[fatal] copy of %s differs from its source" % src)
        return "copied"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(SRC, "pairs_manifest.json")) as f:
        source = json.load(f)
    counts, how, placed = {}, {"linked": 0, "already": 0, "copied": 0}, set()
    for side in SIDES:
        os.makedirs(os.path.join(DST, side), exist_ok=True)
    for m in MAPS:
        names = {s: sorted(f for f in os.listdir(os.path.join(SRC, m, s)) if f.endswith(".png")) for s in SIDES}
        if names["HR"] != names["LRx4"]:
            raise SystemExit("[fatal] %s: HR and LRx4 hold different names" % m)
        bad = [n for n in names["HR"] if not n.endswith("__%s.png" % m)]
        if bad:
            raise SystemExit("[fatal] %s: names without the __%s suffix, e.g. %s" % (m, m, bad[0]))
        for s in SIDES:
            for n in names[s]:
                how[place(os.path.join(SRC, m, s, n), os.path.join(DST, s, n), args.force)] += 1
        counts[m] = len(names["HR"])
        placed.update(names["HR"])
        print("  %-10s %5d pairs" % (m, counts[m]))

    # Anything else in the folders (an older run's leftovers) would be trained on too.
    extra = {s: sorted(set(os.listdir(os.path.join(DST, s))) - placed) for s in SIDES}
    manifest = {"source": os.path.relpath(SRC, paths.REPO), "pairs": sum(counts.values()), "per_map": counts,
                "placed": how, "materials": len(source["kept"]),
                "excluded_by_source": source["excluded"], "constant_maps": source["constant_maps"],
                "recipe": source["recipe"], "unexpected_files": extra}
    with open(os.path.join(DST, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("%d pairs (linked %d, already linked %d, copied %d) -> %s"
          % (manifest["pairs"], how["linked"], how["already"], how["copied"], DST))
    if any(extra.values()):
        raise SystemExit("[fatal] files in %s that no source map has: %s" % (DST, {s: v[:3] for s, v in extra.items()}))


if __name__ == "__main__":
    main()
