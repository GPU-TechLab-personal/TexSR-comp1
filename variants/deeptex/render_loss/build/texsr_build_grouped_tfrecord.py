#!/usr/bin/env python3
"""Pack the MatSynth PNG pairs into a GROUPED TFRecord: one Example = one material.

Stage prep-grouped, step 1 of 2 (texsr_verify_grouped.py is step 2). Runs in the
deeptex container, because it needs tf.io.TFRecordWriter as
texsr_build_tfrecord.py does:

    python /workspace/scripts/deeptex/texsr_build_grouped_tfrecord.py [--force]
    python /workspace/scripts/deeptex/texsr_verify_grouped.py

WHY A SECOND RECORD. The render-space loss (render_tf.py) renders all five maps
of one material under one crop, so a batch has to carry the five maps together.
The paper-path record (texsr_build_tfrecord.py) holds one map per Example, and
upstream crops each Example on its own (modules/dataset.py:52-81), so no batch
from it can ever be rendered. Here each Example holds the ten PNG strings of one
material, under grouped_dataset.SCHEMA:

    image/img_name     <safe>, the material name (not a file name)
    image/hr/<map>     bytes of <root>/<map>/HR/<safe>__<map>.png
    image/lr/<map>     bytes of <root>/<map>/LRx4/<safe>__<map>.png

for <map> in baseColor, normal, roughness, metallic, height. The path rule is
matsynth_make_pairs.py:157-159 (out_paths). That file cannot be imported here,
because it imports pyarrow at :72. The bytes are copied and never re-encoded, so
every pixel is one matsynth_all trained on, and texsr_verify_grouped.py can prove
the packing by plain byte comparison.

WHERE THE NAMES COME FROM. The list is _meta/<safe>.json["safe"]. The pair maker
writes that sidecar LAST (matsynth_make_pairs.py:255-257), so a sidecar means the
material is complete. The ten folder listings are checked against that list with
rsplit("__", 1). A plain split would be wrong because blends are named
ms_<a>__<b>, so the material part can itself contain "__". Every file must also
end in exactly "__<map>.png". Without that rule a leftover
x__baseColor.partial.png would split into the valid name "x". If any of this
disagrees, the build stops before the writer opens, as texsr_build_tfrecord.py
does for missing partners.

ORDER. The names are sorted, then shuffled with random.Random(seed), as in
texsr_build_tfrecord.py:142-143. The render arms train with --shuffle off, so this
order IS the curriculum, and R0a, R0b and R1 all read it the same way. --limit N
keeps only the first N names of that order, which gives a prefix of the full
curriculum for quick tests. texsr_verify_grouped.py never passes such a record:
the best it can get is PARTIAL, and deeptex_train.py refuses that.

STALE STATE. Before the writer opens, three files are deleted:
<out>.meta.json, <dirname(out)>/verify_report.json and any old .partial. The
report check in check_dataset (deeptex_train.py) accepts any PASS report beside
the record, so a report left from the previous build would vouch for bytes it
never read. With both files gone, a crash anywhere below leaves a record the
trainer refuses (it has no meta) until build and verify have both finished. One
record per folder is enforced for the same reason: deeptex_train.py reads
dirname(record)/verify_report.json, so two records in one folder would share a
single report.

META <out>.meta.json, and who reads which keys:
  record_count, hr_size, lr_size   grouped_dataset.load_grouped_dataset (crop
                                   limit, resume skip)
  schema, sha256, bytes            deeptex_train.py check_dataset and
                                   check_grouped_dataset
  seed, limit, root, order_sha256  texsr_verify_grouped.py, which recomputes the
                                   order from them
order_sha256 is the sha256 of "\\n".join(img_name in record order), UTF-8.

Exit codes: 0 ok; 2 bad arguments, or --out under external/ or beside another
record; 3 --out exists (pass --force); 4 the pair folders are inconsistent;
5 the written record failed its decode check (no meta is written, so the trainer
refuses it).
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import random
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4)))
from variants.common import paths  # noqa: E402

import tensorflow as tf  # noqa: E402

from variants.deeptex.render_loss import grouped_dataset as gd  # noqa: E402
from variants.deeptex.upstream_runner.texsr_build_tfrecord import _bytes_feature, sha256  # noqa: E402,E501

DEFAULT_ROOT = "/workspace/runs/deeptex/datasets/matsynth"
DEFAULT_OUT = ("/workspace/runs/deeptex/datasets/matsynth_grouped/"
               "matsynth_grouped_lrx4.tfrecord")
REPO = paths.REPO

# (schema side, pair sub-folder, side length): the HR strings first, then LR, as
# in grouped_dataset.SCHEMA.
SIDES = (("hr", "HR", gd.HR_SIZE), ("lr", "LRx4", gd.LR_SIZE))
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def die(code, msg):
    print("[fatal] %s" % msg)
    sys.exit(code)


def cpu_only():
    """Nothing here needs the GPU. Hiding it stops TF from reserving all of the
    GPU's memory when the context starts. Must run before the first TF op."""
    try:
        tf.config.set_visible_devices([], "GPU")
    except (RuntimeError, ValueError):
        pass


def under_external(path):
    """True if `path` lies inside the repo's external/ (the two submodules).

    The deeptex container's working directory IS external/DeepTextures, so a
    relative path would land inside the submodule.
    """
    ext = os.path.realpath(os.path.join(REPO, "external"))
    p = os.path.realpath(os.path.abspath(path))
    return p == ext or p.startswith(ext + os.sep)


def pair_path(root, name, key, sub):
    """<root>/<map>/<HR|LRx4>/<safe>__<map>.png (matsynth_make_pairs.py:157-159)."""
    return os.path.join(root, key, sub, "%s__%s.png" % (name, key))


def schema_files(root, name):
    """[(schema key, pair file, side length)] for the ten PNGs of one material,
    in grouped_dataset.SCHEMA order."""
    return [("image/%s/%s" % (side, key), pair_path(root, name, key, sub), size)
            for side, sub, size in SIDES for key in gd.MAP_KEYS]


def png_ihdr(b):
    """(width, height, bit depth, colour type) read from the PNG header, or None.

    The PNG spec puts IHDR first: signature [0:8], chunk length [8:12], b"IHDR"
    [12:16], width [16:20], height [20:24], bit depth [24], colour type [25].
    """
    if len(b) < 26 or b[:8] != PNG_SIGNATURE or b[12:16] != b"IHDR":
        return None
    return (int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big"),
            b[24], b[25])


def png_problem(b, size):
    """None if `b` starts an 8-bit RGB size x size PNG, else what is wrong with it.

    Colour type 2 is truecolour without alpha. decode_png(channels=3)
    (modules/dataset.py:12-13) would silently coerce grey (0), palette (3) and
    RGBA (6), and turn 16-bit into 8-bit.
    """
    hdr = png_ihdr(b)
    if hdr is None:
        return "not a PNG (no signature / IHDR)"
    w, h, depth, ctype = hdr
    if depth != 8 or ctype != 2:
        return "bit depth %d, colour type %d (want 8-bit RGB: 8, 2)" % (depth, ctype)
    if (w, h) != (size, size):
        return "%dx%d (want %dx%d)" % (w, h, size, size)
    return None


def brief(items, k=8):
    """The first k of `items`, sorted, as one line for an error message."""
    items = sorted(items)
    return ", ".join(items[:k]) + (" ..." if len(items) > k else "")


def read_sidecars(root):
    """Sorted material names, from <root>/_meta/*.json["safe"].

    Raises ValueError if any sidecar is unreadable, is not named <safe>.json, or
    lists fewer than the five maps.
    """
    paths = sorted(glob.glob(os.path.join(root, "_meta", "*.json")))
    if not paths:
        raise ValueError("no sidecars under %s - run matsynth_make_pairs.py"
                         % os.path.join(root, "_meta"))
    names, bad = [], []
    for p in paths:
        base = os.path.basename(p)
        try:
            with open(p) as f:
                rec = json.load(f)
        except (OSError, ValueError) as e:
            # json.dump at matsynth_make_pairs.py:256-257 is not atomic, so a
            # killed pair run can leave a truncated sidecar behind.
            bad.append("%s: unreadable (%s)" % (base, e))
            continue
        safe = rec.get("safe") if isinstance(rec, dict) else None
        if not isinstance(safe, str) or not safe:
            bad.append("%s: no 'safe' name" % base)
            continue
        if safe + ".json" != base:
            bad.append("%s: holds safe=%r" % (base, safe))
            continue
        maps = rec.get("maps")
        have = set(maps) if isinstance(maps, (dict, list)) else set()
        missing = [k for k in gd.MAP_KEYS if k not in have]
        if missing:
            bad.append("%s: no %s in 'maps'" % (base, "/".join(missing)))
            continue
        names.append(safe)
    if bad:
        raise ValueError("%d bad sidecar(s) under %s/_meta: %s"
                         % (len(bad), root, "; ".join(bad[:8])))
    return sorted(names)


def folder_names(root, key, sub):
    """Material names in one of the ten pair folders, by rsplit("__", 1).

    Raises ValueError on any file not named exactly <material>__<key>.png.
    """
    d = os.path.join(root, key, sub)
    if not os.path.isdir(d):
        raise ValueError("no such folder: %s" % d)
    tail = key + ".png"
    names, odd = set(), []
    for f in os.listdir(d):
        parts = f.rsplit("__", 1)
        if len(parts) != 2 or not parts[0] or parts[1] != tail:
            odd.append(f)
        else:
            names.add(parts[0])
    if odd:
        raise ValueError("%s: %d file(s) not named <material>__%s: %s"
                         % (d, len(odd), tail, brief(odd, 5)))
    return names


def list_materials(root):
    """The sidecar names, after checking that all ten folders hold exactly them."""
    names = read_sidecars(root)
    want = set(names)
    diffs = []
    for key in gd.MAP_KEYS:
        for _, sub, _ in SIDES:
            got = folder_names(root, key, sub)
            if got != want:
                diffs.append("%s/%s: %d file(s) without a sidecar [%s], %d sidecar(s) "
                             "without a file [%s]"
                             % (key, sub, len(got - want), brief(got - want, 3),
                                len(want - got), brief(want - got, 3)))
    if diffs:
        raise ValueError("the pair folders disagree with _meta (an interrupted "
                         "matsynth_make_pairs.py run? rerun it to completion):\n  "
                         + "\n  ".join(diffs))
    return names


def check_manifest(root, n):
    """Tie the build to a COMPLETED pair run. The pair maker writes
    pairs_manifest.json only after its file counts check out, so a manifest
    whose kept count disagrees with the sidecars is left over from another run.

    Returns the manifest path, or "" if there is none.
    """
    p = os.path.join(root, "pairs_manifest.json")
    if not os.path.isfile(p):
        print("[warn] no %s; building from the sidecars alone" % p)
        return ""
    with open(p) as f:
        m = json.load(f)
    kept = (m.get("materials") or {}).get("kept")
    if kept != n:
        raise ValueError("%s says %s material(s) kept, but _meta holds %d"
                         % (p, kept, n))
    for k, want in (("hr_size", gd.HR_SIZE), ("lr_size", gd.LR_SIZE)):
        if m.get(k, want) != want:
            raise ValueError("%s has %s=%s; the grouped record is %d/%d"
                             % (p, k, m.get(k), gd.HR_SIZE, gd.LR_SIZE))
    return p


def material_order(names, seed, limit=0):
    """sorted(names), then random.Random(seed).shuffle (texsr_build_tfrecord.py:142-143);
    the first `limit` if > 0. texsr_verify_grouped.py calls this to recompute
    the order, so the two cannot disagree."""
    order = sorted(names)
    random.Random(seed).shuffle(order)
    return order[:limit] if limit and limit > 0 else order


def order_sha256(names):
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def prescan(root, names):
    """Read the header of all ten PNGs of every material BEFORE anything is
    deleted or written. A bad file found mid-write would leave a half-built
    record, and by then the old meta would already be gone."""
    bad = []
    for name in names:
        for _, path, size in schema_files(root, name):
            with open(path, "rb") as f:
                why = png_problem(f.read(32), size)
            if why:
                bad.append("%s: %s" % (path, why))
    if bad:
        raise ValueError("%d PNG(s) are not 8-bit RGB at the expected size: %s"
                         % (len(bad), "; ".join(bad[:8])))


def make_grouped_example(name, pngs):
    """pngs: {schema key: PNG bytes} for the ten image/hr/* and image/lr/* keys."""
    feature = {gd.SCHEMA[0]: _bytes_feature(name.encode("utf-8"))}
    for k in gd.SCHEMA[1:]:
        feature[k] = _bytes_feature(pngs[k])
    return tf.train.Example(features=tf.train.Features(feature=feature))


def decode_check(path, order, k):
    """Decode the first k records through the trainer's own parse spec
    (grouped_dataset.feature_spec). A broken record then costs seconds here
    instead of surfacing thousands of steps into training. The full proof is
    texsr_verify_grouped.py."""
    spec = gd.feature_spec()
    problems, checked = [], 0
    for i, raw in enumerate(tf.data.TFRecordDataset(path).take(k)):
        try:
            ex = tf.io.parse_single_example(raw, spec)
            name = ex[gd.SCHEMA[0]].numpy().decode("utf-8")
            if name != order[i]:
                problems.append("record %d is %r, expected %r" % (i, name, order[i]))
            for key, _, size in schema_files("", name):
                img = tf.image.decode_png(ex[key], channels=0)
                if tuple(img.shape) != (size, size, 3) or img.dtype != tf.uint8:
                    problems.append("%s %s decodes to %s %s"
                                    % (name, key, tuple(img.shape), img.dtype.name))
        except (tf.errors.OpError, UnicodeDecodeError) as e:
            problems.append("record %d: %s" % (i, (str(e).splitlines() or [""])[0]))
        checked += 1
    return checked, problems


def write_json_atomic(path, obj):
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="pair root written by matsynth_make_pairs.py")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output .tfrecord; its folder gets the meta and the verify report")
    ap.add_argument("--seed", type=int, default=0,
                    help="record-order seed (the curriculum under --shuffle off)")
    ap.add_argument("--limit", type=int, default=0,
                    help="pack only the first N materials of the order (tests; "
                         "texsr_verify_grouped.py rates such a record PARTIAL)")
    ap.add_argument("--sample", type=int, default=16,
                    help="decode-check this many records after writing")
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    args = ap.parse_args()

    cpu_only()
    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out)
    out_dir = os.path.dirname(out)
    if under_external(out):
        die(2, "--out %s is inside external/ (the container's working directory is "
               "the submodule; give an absolute /workspace/runs/... path)" % out)
    if args.limit < 0 or args.sample < 0:
        die(2, "--limit and --sample must be >= 0")
    if os.path.isdir(out_dir):
        others = [p for p in glob.glob(os.path.join(out_dir, "*.tfrecord"))
                  if os.path.abspath(p) != out]
        if others:
            die(2, "%s already holds %s. deeptex_train.py reads ONE verify_report.json "
                   "per folder, so a grouped record needs a folder of its own"
                   % (out_dir, brief([os.path.basename(p) for p in others], 3)))
    if os.path.exists(out) and not args.force:
        die(3, "%s already exists (pass --force to rebuild it)" % out)
    if not os.path.isdir(root):
        die(4, "no pair root %s - run matsynth_make_pairs.py first" % root)
    if [k for k, _, _ in schema_files("", "x")] != list(gd.SCHEMA[1:]):
        die(4, "schema_files() is out of step with grouped_dataset.SCHEMA")

    t0 = time.time()
    try:
        names = list_materials(root)
        manifest = check_manifest(root, len(names))
    except ValueError as e:
        die(4, str(e))
    order = material_order(names, args.seed, args.limit)
    print("[*] %d material(s) in %s; packing %d (seed %d%s)"
          % (len(names), root, len(order), args.seed,
             ", --limit %d" % args.limit if args.limit else ""))
    try:
        prescan(root, order)
    except ValueError as e:
        die(4, str(e))
    print("[*] %d PNG headers ok: 8-bit RGB, HR %d^2, LR %d^2   (%.0f s)"
          % (len(order) * 2 * gd.N_MAPS, gd.HR_SIZE, gd.LR_SIZE, time.time() - t0))

    # Stale state goes BEFORE the writer opens; see the module docstring.
    os.makedirs(out_dir, exist_ok=True)
    meta_path = out + ".meta.json"
    tmp = out + ".partial"
    for stale in (os.path.join(out_dir, "verify_report.json"), meta_path,
                  meta_path + ".partial", tmp):
        if os.path.exists(stale):
            os.remove(stale)
            print("[*] removed %s" % stale)

    n = 0
    try:
        with tf.io.TFRecordWriter(tmp) as w:
            for name in order:
                pngs = {}
                for key, path, _ in schema_files(root, name):
                    with open(path, "rb") as f:
                        pngs[key] = f.read()
                w.write(make_grouped_example(name, pngs).SerializeToString())
                n += 1
                if n % 200 == 0:
                    print("  %d/%d   (%.0f s)" % (n, len(order), time.time() - t0),
                          flush=True)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, out)                                  # texsr_build_tfrecord.py:145-158
    print("[*] wrote %d material(s) = %d PNG strings -> %s (%.2f GB)"
          % (n, n * 2 * gd.N_MAPS, out, os.path.getsize(out) / 1e9))

    checked, problems = decode_check(out, order, args.sample)
    if problems:
        for p in problems[:20]:
            print("[FAIL] %s" % p)
        die(5, "%d problem(s) in the first %d record(s); no meta written, so "
               "deeptex_train.py will refuse %s" % (len(problems), checked, out))
    print("[*] decode-checked %d record(s) through grouped_dataset.feature_spec()" % checked)

    meta = {
        "tfrecord": out,
        "record_count": n,
        "seed": args.seed,
        "limit": args.limit,
        "materials_available": len(names),
        "sha256": sha256(out),
        "bytes": os.path.getsize(out),
        "maps": list(gd.MAP_KEYS),
        "schema": list(gd.SCHEMA),
        "root": root,
        "pairs_manifest": manifest,
        "pairs_manifest_sha256": sha256(manifest) if manifest else "",
        "hr_size": gd.HR_SIZE,
        "lr_size": gd.LR_SIZE,
        "scale": gd.HR_SIZE // gd.LR_SIZE,
        "order_sha256": order_sha256(order),
        "created": datetime.datetime.now(datetime.timezone.utc)
                   .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": os.path.basename(__file__),
    }
    write_json_atomic(meta_path, meta)
    print("[build] records=%d sha256=%s... order_sha256=%s... seconds=%.0f"
          % (n, meta["sha256"][:16], meta["order_sha256"][:16], time.time() - t0))
    print("meta -> %s" % meta_path)
    print("next: python /workspace/scripts/deeptex/texsr_verify_grouped.py --tfrecord %s"
          % out)


if __name__ == "__main__":
    main()
