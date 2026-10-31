#!/usr/bin/env python3
"""Prove the grouped TFRecord is exactly the PNG pairs it claims to pack.

Stage prep-grouped, step 2 of 2, after texsr_build_grouped_tfrecord.py. Runs in
the deeptex container. It writes <dirname(tfrecord)>/verify_report.json.
deeptex_train.py accepts a grouped record only if two things hold: the report's
verdict is PASS (check_dataset), and its "meta" block (sha256, bytes,
record_count, measured HERE from the file that was read) equals the record's own
meta (check_grouped_dataset). So a report vouches for the exact bytes it
verified and for nothing else.

WHAT IS CHECKED, and why each check is needed:

  meta        <record>.meta.json exists, and its record_count, bytes and sha256
              match the file. A truncated TFRecord does not raise on open:
              TFRecordDataset just ends early, and repeat() hides that. Its
              schema, maps, hr_size and lr_size must be the values that
              grouped_dataset.py and deeptex_train.py read.
  keys        Every Example holds exactly grouped_dataset.SCHEMA, with one byte
              string per key. The raw protobuf is inspected, because
              parse_single_example only checks that the spec's keys are PRESENT
              and would accept extra keys.
  names       Names are unique and equal the _meta sidecar set of --root (2,904
              materials), so no material is lost or packed twice.
  order       The record order equals material_order(_meta names, meta.seed,
              meta.limit), the builder's own function, and hashes to
              meta.order_sha256. Under --shuffle off this order is the
              curriculum that R0a, R0b and R1 have to share.
  PNG header  On every string, IHDR byte 24 (bit depth) == 8 and byte 25 (colour
              type) == 2 (RGB). texsr_verify_tfrecord.py:100-103 tests the dtype
              of decode_png(channels=0), but decode_png returns the REQUESTED
              dtype (uint8 by default), so that test cannot see a 16-bit file.
              The header byte can. channels=3 (modules/dataset.py:12-13) would
              also coerce grey, palette and RGBA files silently.
  decode      decode_png(channels=0) gives (512, 512, 3) for HR and
              (128, 128, 3) for LR.
  bytes       Each string is byte-identical to
              <root>/<map>/{HR,LRx4}/<name>__<map>.png. This is the check that
              proves no map sits in another map's slot and no material carries
              another material's file: 29,040 comparisons.
  GT decode   render_tf.decode_maps of each material's five HR maps is finite.
              It always should be. A failure here means the renderer port and the
              data disagree before training has even started.
INFORMATIONAL (reported, never failed):
  grey maps   roughness, metallic and height have three equal channels.
              render_tf takes the channel mean, which equals MatUp's channel-0
              read only when the channels are equal.
  normal_invalid  For each material, the fraction of GT HR normal texels that
              the render loss masks out: |2v-1| < 0.1 or unit n_z < 1e-3. It is
              computed by render_tf.decode_maps itself (its `valid` output), so
              the number is the training rule and not a re-implementation of it.
              Materials whose "normal" is not a normal map (a constant colour,
              say) show up at 100%.

VERDICT. PASS (exit 0) requires a complete check (--sample 0) of a full build
(no --limit) with no failure. Any failure gives FAIL (exit 1). A clean sampled
check, or a clean --limit record, gives PARTIAL (exit 3). PARTIAL is useful
while developing, but deeptex_train.py accepts only PASS. Exit 2 means bad
arguments: a missing record, or a report path inside external/.

The GPU is hidden. Nothing here needs it, and TF would otherwise reserve all of
the card's memory. The mask is a threshold test on values derived from uint8
pixels, so the CPU and the GPU could only disagree on a texel that lies within
float32 rounding of a threshold. The statistic is informational either way.
"""

import argparse
import collections
import datetime
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4)))
from variants.common import paths  # noqa: E402

import tensorflow as tf  # noqa: E402

from variants.deeptex.render_loss import grouped_dataset as gd  # noqa: E402
from variants.deeptex.render_loss import render_tf  # noqa: E402
from variants.deeptex.render_loss.build import texsr_build_grouped_tfrecord as build  # noqa: E402,E501  (its main() is guarded)

GREY_KEYS = ("roughness", "metallic", "height")
DECODED_KEYS = ("albedo", "normal", "alpha", "metallic", "pos")


class Failures(object):
    """Counts every failure and keeps the first CAP of them verbatim, so 29,040
    byte mismatches do not become a 29,040-line report."""
    CAP = 200

    def __init__(self):
        self.count = 0
        self.kept = []

    def add(self, msg):
        self.count += 1
        if len(self.kept) < self.CAP:
            self.kept.append(msg)


def gt_mask_counter(hr_side):
    """tf.function: one material's HR stack (5, S, S, 3) uint8 ->
    (texels masked out, texels the n_z guard moved, every decoded value finite).

    `hr5 / 255` is the trainer's exact conversion (crop_and_augment's uint8
    truediv, which yields float32); decode_maps is the trainer's exact decode.
    The counts are float32 sums of 0/1 values, which stay exact up to 2^24 (512^2
    is 2^18).
    """
    pos = tf.constant(gd.position_grid_np(hr_side)[None])

    @tf.function(input_signature=[tf.TensorSpec([gd.N_MAPS, hr_side, hr_side, 3], tf.uint8)])
    def count(hr5):
        d = render_tf.decode_maps((hr5 / 255)[None], pos)
        finite = tf.reduce_all(tf.stack(
            [tf.reduce_all(tf.math.is_finite(d[k])) for k in DECODED_KEYS]))
        return tf.reduce_sum(1.0 - d["valid"]), tf.reduce_sum(d["nz_guarded"]), finite
    return count


def check_strings(name, vals, root, fails, png):
    """Header, disk bytes and decode of one record's ten PNG strings.
    Returns {schema key: decoded uint8 tensor} for the strings that decoded to
    the right shape."""
    decoded = {}
    for key, path, size in build.schema_files(root, name):
        data = vals.get(key)
        if data is None:
            continue                      # already failed under "keys"
        png["strings"] += 1
        why = build.png_problem(data, size)
        if why:
            png["header_bad"] += 1
            fails.add("%s %s: %s" % (name, key, why))
        try:
            with open(path, "rb") as f:
                disk = f.read()
        except OSError:
            png["file_missing"] += 1
            fails.add("%s %s: no pair file %s" % (name, key, path))
        else:
            if disk != data:
                png["bytes_mismatch"] += 1
                fails.add("%s %s: differs from %s (%d vs %d bytes)"
                          % (name, key, path, len(data), len(disk)))
        try:
            img = tf.image.decode_png(data, channels=0)
        except tf.errors.OpError as e:
            png["decode_bad"] += 1
            fails.add("%s %s: decode failed: %s"
                      % (name, key, (str(e).splitlines() or [""])[0]))
            continue
        if tuple(img.shape) != (size, size, 3) or img.dtype != tf.uint8:
            png["shape_bad"] += 1
            fails.add("%s %s: decodes to %s %s, want (%d, %d, 3) uint8"
                      % (name, key, tuple(img.shape), img.dtype.name, size, size))
            continue
        decoded[key] = img
    return decoded


def grey_unequal(name, decoded):
    out = []
    for side in ("hr", "lr"):
        for k in GREY_KEYS:
            img = decoded.get("image/%s/%s" % (side, k))
            if img is None:
                continue
            a = img.numpy()
            if not ((a[..., 0] == a[..., 1]).all() and (a[..., 0] == a[..., 2]).all()):
                out.append("%s %s/%s" % (name, side, k))
    return out


def write_json_atomic(path, obj):
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tfrecord", default=build.DEFAULT_OUT)
    ap.add_argument("--root", default=build.DEFAULT_ROOT,
                    help="pair root the record must match byte for byte (the canonical "
                         "one by default, deliberately NOT the meta's own root)")
    ap.add_argument("--sample", type=int, default=0,
                    help="run the per-string checks on the first N records only "
                         "(0 = all). Anything but a complete check is at best PARTIAL")
    args = ap.parse_args()

    build.cpu_only()
    path = os.path.abspath(args.tfrecord)
    root = os.path.abspath(args.root)
    report_path = os.path.join(os.path.dirname(path), "verify_report.json")
    if build.under_external(report_path):
        build.die(2, "%s is inside external/; refusing to write a report there" % report_path)
    if not os.path.isfile(path):
        build.die(2, "no such file: %s - run texsr_build_grouped_tfrecord.py" % path)
    if args.sample < 0:
        build.die(2, "--sample must be >= 0")

    t0 = time.time()
    fails = Failures()
    warnings = []
    checks = collections.OrderedDict()

    # ---- meta (read first; the order check below needs its seed and limit)
    meta_path = path + ".meta.json"
    meta = None
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except ValueError as e:
            fails.add("%s is not valid JSON: %s" % (meta_path, e))
    else:
        fails.add("no %s - the build did not finish (texsr_build_grouped_tfrecord.py "
                  "writes it last)" % meta_path)

    # ---- the sidecar set the names must equal
    try:
        sidecars = build.read_sidecars(root)
    except ValueError as e:
        sidecars = None
        fails.add(str(e))

    # ---- one pass over the records
    hr_side, lr_side = gd.HR_SIZE, gd.LR_SIZE
    schema_set = set(gd.SCHEMA)
    counter = gt_mask_counter(hr_side)
    names, bad_keys, deep = [], 0, 0
    png = collections.OrderedDict((k, 0) for k in (
        "strings", "header_bad", "decode_bad", "shape_bad", "bytes_mismatch", "file_missing"))
    grey_bad, per_mat = [], []
    stream_ok = True
    n = 0
    it = iter(tf.data.TFRecordDataset(path))
    while True:
        try:
            raw = next(it)
        except StopIteration:
            break
        except tf.errors.OpError as e:
            stream_ok = False
            fails.add("record %d: the TFRecord stream broke (%s) - truncated or corrupt"
                      % (n + 1, (str(e).splitlines() or [""])[0]))
            break
        n += 1
        try:
            ex = tf.train.Example.FromString(raw.numpy())
        except Exception as e:  # noqa: BLE001 - want the record index in the message
            bad_keys += 1
            fails.add("record %d: not a tf.train.Example (%s)" % (n, e))
            names.append("")
            continue

        feats = ex.features.feature
        keys = set(feats.keys())
        if keys != schema_set:
            bad_keys += 1
            fails.add("record %d: keys differ from SCHEMA (missing %s, extra %s)"
                      % (n, sorted(schema_set - keys), sorted(keys - schema_set)))
        vals = {}
        for k in gd.SCHEMA:
            if k not in keys:
                continue
            f = feats[k]
            if f.WhichOneof("kind") != "bytes_list" or len(f.bytes_list.value) != 1:
                bad_keys += 1
                fails.add("record %d: %s is not exactly one byte string" % (n, k))
                continue
            vals[k] = f.bytes_list.value[0]
        name = vals.get(gd.SCHEMA[0], b"").decode("utf-8", "replace")
        names.append(name)
        if not name:
            fails.add("record %d: empty img_name" % n)
            continue
        if args.sample and n > args.sample:
            continue

        deep += 1
        decoded = check_strings(name, vals, root, fails, png)
        grey_bad += grey_unequal(name, decoded)
        hr_keys = ["image/hr/" + k for k in gd.MAP_KEYS]
        if all(k in decoded for k in hr_keys):
            inv, guarded, finite = counter(tf.stack([decoded[k] for k in hr_keys]))
            if not bool(finite):
                fails.add("%s: render_tf.decode_maps gives a non-finite value on the GT maps"
                          % name)
            per_mat.append((name, float(inv), float(guarded)))
        if deep % 250 == 0:
            print("  %d record(s) checked   (%.0f s)" % (deep, time.time() - t0), flush=True)

    print("records: %d (%d with per-string checks)" % (n, deep))
    if n == 0:
        fails.add("0 records: the grouped loader's repeat() would yield nothing")
    checks["stream"] = {"records": n, "read_to_end": stream_ok}
    checks["keys"] = {"schema": list(gd.SCHEMA), "records_bad": bad_keys,
                      "ok": bad_keys == 0}

    # ---- file vs meta
    digest = build.sha256(path)
    size = os.path.getsize(path)
    m = meta or {}
    if meta is not None:
        for k, got in (("record_count", n), ("bytes", size), ("sha256", digest)):
            if m.get(k) != got:
                fails.add("%s mismatch: meta says %r, file has %r" % (k, m.get(k), got))
        for k, want in (("schema", list(gd.SCHEMA)), ("maps", list(gd.MAP_KEYS)),
                        ("hr_size", hr_side), ("lr_size", lr_side)):
            if m.get(k) != want:
                fails.add("meta %s is %r, the grouped loader expects %r" % (k, m.get(k), want))
        if os.path.normpath(str(m.get("root"))) != root:
            fails.add("the record was built from %s but is being verified against %s "
                      "(pass --root to match, or rebuild)" % (m.get("root"), root))
    checks["meta"] = collections.OrderedDict([
        ("present", meta is not None),
        ("record_count", {"file": n, "meta": m.get("record_count")}),
        ("bytes", {"file": size, "meta": m.get("bytes")}),
        ("sha256", {"file": digest, "meta": m.get("sha256")}),
    ])

    # ---- names and order
    counts = collections.Counter(nm for nm in names if nm)
    dupes = sorted(nm for nm, c in counts.items() if c > 1)
    for nm in dupes[:20]:
        fails.add("duplicate img_name %r (%d records)" % (nm, counts[nm]))
    if len(dupes) > 20:
        fails.add("... %d more duplicated names" % (len(dupes) - 20))
    seed, limit = m.get("seed"), m.get("limit") or 0
    expected = None
    if sidecars is not None:
        if meta is not None and isinstance(seed, int) and isinstance(limit, int):
            expected = build.material_order(sidecars, seed, limit)
            want = set(expected)
        else:
            if meta is not None:
                fails.add("meta seed=%r limit=%r: cannot recompute the record order"
                          % (seed, limit))
            want = set(sidecars)
        got = set(counts)
        missing, extra = sorted(want - got), sorted(got - want)
        if missing:
            fails.add("%d material(s) expected from %s/_meta are not in the record: %s"
                      % (len(missing), root, build.brief(missing, 5)))
        if extra:
            fails.add("%d record name(s) are not expected from %s/_meta: %s"
                      % (len(extra), root, build.brief(extra, 5)))
    order_ok = expected is not None and names == expected
    if expected is not None and not order_ok and not dupes and set(names) == set(expected):
        first = next(i for i, (a, b) in enumerate(zip(names, expected)) if a != b)
        fails.add("record order differs from material_order(_meta, seed=%d, limit=%d) "
                  "from record %d on" % (seed, limit, first + 1))
    file_order_sha = build.order_sha256(names)
    if meta is not None and m.get("order_sha256") != file_order_sha:
        fails.add("order_sha256 mismatch: meta %s..., record %s..."
                  % (str(m.get("order_sha256"))[:16], file_order_sha[:16]))
    checks["names"] = collections.OrderedDict([
        ("unique", not dupes),
        ("duplicates", len(dupes)),
        ("sidecars", len(sidecars) if sidecars is not None else None),
        ("expected", len(expected) if expected is not None else None),
        ("in_record", len(counts)),
    ])
    checks["order"] = {"seed": seed, "limit": limit, "matches_material_order": order_ok,
                       "order_sha256": file_order_sha}

    # ---- per-string results
    checks["png"] = collections.OrderedDict(
        [("records_checked", deep)] + list(png.items())
        + [("ok", deep > 0 and all(v == 0 for k, v in png.items() if k != "strings"))])
    checks["grey_maps_equal_channels"] = {"unequal": len(grey_bad),
                                          "examples": grey_bad[:20]}
    if grey_bad:
        warnings.append("%d grey map(s) without three equal channels (e.g. %s): "
                        "render_tf's channel mean differs from MatUp's channel-0 read there"
                        % (len(grey_bad), grey_bad[0]))

    # ---- the GT normal mask, as the render loss will apply it
    rc = render_tf.validate_render_cfg(None, warn=False)
    texels = hr_side * hr_side
    tot = len(per_mat) * texels
    inv_sum = sum(p[1] for p in per_mat)
    flagged = sorted((p for p in per_mat if p[1] > 0), key=lambda p: (-p[1], p[0]))
    normal_invalid = collections.OrderedDict([
        ("rule", "render_tf.decode_maps valid == 0 on the GT HR normal: "
                 "|2v-1| < mask_min_norm or unit n_z < nz_min; these texels are "
                 "masked out of the render loss"),
        ("mask_min_norm", rc["mask_min_norm"]),
        ("nz_min", rc["nz_min"]),
        ("materials_checked", len(per_mat)),
        ("texels_checked", tot),
        ("texels_invalid", int(inv_sum)),
        ("fraction_invalid", inv_sum / tot if tot else None),
        ("texels_nz_guarded", int(sum(p[2] for p in per_mat))),
        ("materials_with_invalid", len(flagged)),
        ("materials_over_1pct", sum(p[1] > 0.01 * texels for p in flagged)),
        ("materials_over_10pct", sum(p[1] > 0.10 * texels for p in flagged)),
        ("materials_all_invalid", [p[0] for p in flagged if p[1] == texels]),
        ("per_material", [collections.OrderedDict([
            ("name", p[0]),
            ("invalid", round(p[1] / texels, 6)),
            ("nz_guarded", round(p[2] / texels, 6))]) for p in flagged]),
    ])

    # ---- verdict
    complete = args.sample == 0 or args.sample >= n
    full_build = not limit
    if fails.count:
        verdict = "FAIL"
    elif complete and full_build:
        verdict = "PASS"
    else:
        verdict = "PARTIAL"
        warnings.append("verdict PARTIAL (%s): deeptex_train.py accepts only PASS"
                        % ", ".join(w for w, c in (("--sample %d" % args.sample, not complete),
                                                   ("--limit %s build" % limit, not full_build))
                                    if c))

    report = collections.OrderedDict([
        ("verdict", verdict),
        ("tfrecord", path),
        ("root", root),
        # Measured from the file this run read; check_grouped_dataset requires
        # these three to equal the record's meta.
        ("meta", collections.OrderedDict([
            ("sha256", digest), ("bytes", size), ("record_count", n)])),
        ("checks", checks),
        ("failure_count", fails.count),
        ("failures", fails.kept),
        ("warnings", warnings),
        ("normal_invalid", normal_invalid),
        ("verifier", os.path.basename(__file__)),
        ("created", datetime.datetime.now(datetime.timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ")),
        ("seconds", round(time.time() - t0, 1)),
    ])
    write_json_atomic(report_path, report)

    print("-" * 60)
    print("normal mask: %d of %d GT texels invalid (%.4f%%) in %d material(s); "
          "%d material(s) > 1%%, all-invalid: %s"
          % (int(inv_sum), tot, 100.0 * inv_sum / tot if tot else 0.0, len(flagged),
             normal_invalid["materials_over_1pct"],
             ", ".join(normal_invalid["materials_all_invalid"]) or "none"))
    for p in flagged[:10]:
        print("  %-40s invalid %.4f  nz_guarded %.4f" % (p[0], p[1] / texels, p[2] / texels))
    for w in warnings:
        print("[warn] %s" % w)
    for f_ in fails.kept[:40]:
        print("[FAIL] %s" % f_)
    if fails.count > 40:
        print("[FAIL] ... and %d more (%d kept in the report)"
              % (fails.count - 40, len(fails.kept)))
    print("[verify] verdict=%s records=%d strings=%d failures=%d sha256=%s... seconds=%.0f"
          % (verdict, n, png["strings"], fails.count, digest[:16], time.time() - t0))
    print("report -> %s" % report_path)
    sys.exit({"PASS": 0, "FAIL": 1}.get(verdict, 3))


if __name__ == "__main__":
    main()
