"""Grouped MatSynth loader: one TFRecord Example = all five maps of one material.

Runs in the deeptex container. Used by deeptex_train.py in grouped mode
(train_dataset.grouped: True), by render_gate.py and render_calibrate.py, and its
SCHEMA by texsr_build_grouped_tfrecord.py / texsr_verify_grouped.py.

WHY GROUPED. A render-space loss needs all five SR maps of the SAME material and
the SAME crop at once. The paper-path record holds one map per Example and crops
each independently (external/DeepTextures/modules/dataset.py:52-81), so no batch
ever contains a renderable set. Here every Example carries the ten PNG strings
(5 maps x HR/LR) of one material, and one crop + one flip/rot is drawn per
material and applied to all five maps and to the position grid.

UPSTREAM SEMANTICS ARE KEPT, line by line (modules/dataset.py):
  :53-57  offset = uniform[0, int32.max) % (lr_side - lr_crop + 1); LR cut at the
          offset, HR at offset * scale. Upstream draws a third (channel) component
          whose limit is 1, so only y and x matter.
  :59-66  flip_left_right iff flip_case == 0, flip_case ~ U{0,1}.
  :68-81  rot_case ~ U{0..3}: 0 -> rot90 k=1, 1 -> k=2, 2 -> k=3, 3 -> identity,
          i.e. k = (rot_case + 1) % 4. Flip FIRST, then rotate.
  :83-85  /255 last, after augmentation (uint8 / int -> float32 by truediv).
  :95-104 TFRecordDataset -> repeat -> (shuffle) -> map(AUTOTUNE) ->
          batch(drop_remainder) -> prefetch. Upstream's buffer is 10240; here it
          is capped by bytes (SHUFFLE_BUFFER_BYTES, see WHAT IS DIFFERENT).
The augmentation stays NORMAL-BLIND, as upstream's is
(docs/DEEPTEX-DEVIATIONS.md:389): pixels move, normal components do not. The
render target is unaffected: shading is strictly per texel with a fixed (0,0,1)
view, and the position grid
gets the same flip/rot as the maps, so the GT render of an augmented crop is an
exact permutation of the GT render of the un-augmented crop.

WHAT IS DIFFERENT, and why:
  * Randomness is STATELESS, keyed on (seed, global sample index) via
    Dataset.enumerate(). Upstream uses stateful tf.random.uniform inside an
    AUTOTUNE map, whose assignment of draws to elements is not reproducible.
    Here R0 and R1 with the same seed see the identical crop sequence, and a
    resumed run sees the crops the uninterrupted run would have seen.
  * Resume skips the records already consumed (start_index) instead of
    restarting at record 0 (docs/DEEPTEX-DEVIATIONS.md:763-766 records the
    upstream behaviour). With shuffle off the stream is periodic in the record count, so
    only start_index % record_count records are actually read past. With shuffle
    on all start_index records are read (the shuffled stream is not periodic).
  * The shuffle buffer is min(10240, record_count, 2 GiB of records): a grouped
    Example is ten PNGs, and 10240 of them (~12.7 GB) would not fit in memory.
  * Every element also yields `pos`: the MatUp position grid of the FULL 512^2
    material (pixel centres over [-1,1]^2, data_utils.py:10-16, cast to float32
    as matup_render.position3d does) cut at the HR offset and given the same
    flip/rot as the maps. The renderer's light vectors depend on it.

Nothing here creates a tensor at import time (deeptex_train.py calls
set_memory_growth() before importing this module, and must stay able to).
"""

import json
import os

import numpy as np
import tensorflow as tf

MAP_KEYS = ("baseColor", "normal", "roughness", "metallic", "height")
N_MAPS = 5
SCHEMA = (["image/img_name"]
          + ["image/hr/" + k for k in MAP_KEYS]
          + ["image/lr/" + k for k in MAP_KEYS])

# What texsr_verify_grouped.py enforces on every PNG; used when the record has
# no meta sidecar to read them from.
HR_SIZE = 512
LR_SIZE = 128

# modules/utils.py:38 / modules/dataset.py:93 default buffer.
SHUFFLE_BUFFER = 10240
# The shuffle sits before map, so its buffer holds SERIALIZED Examples, and a
# grouped one carries ten PNGs (~1.24 MB for MatSynth). 10240 of them is ~12.7 GB,
# which the 15.5 GiB docker VM cannot hold beside TF. shuffle_buffer_size() caps
# the buffer at this many bytes of records and at one pass of the record.
SHUFFLE_BUFFER_BYTES = 2 * 1024 ** 3


def feature_spec():
    """parse_single_example spec for SCHEMA: eleven scalar byte strings."""
    return {k: tf.io.FixedLenFeature([], tf.string) for k in SCHEMA}


def position_grid_np(full_size=HR_SIZE):
    """(S, S, 2) float32 MatUp XY grid: [..., 0] = X along columns, [..., 1] = Y
    along rows, pixel centres over [-1, 1].

    data_utils.py:10-16 compute_2D_position_grid in float64, then the float32
    cast of matup_render.py:81 (`.float()`).
    """
    s = int(full_size)
    range_vec = 2 * (np.arange(s) / s + 1 / (2 * s)) - 1             # :11
    x, y = np.meshgrid(range_vec, range_vec)                          # :12
    return np.stack([x, y], axis=-1).astype(np.float32)               # :13-15 + .float()


def augment(x, flip, rot_k):
    """flip_left_right if `flip`, THEN rot90 by `rot_k` (modules/dataset.py:59-81).

    x is a 3-D (H, W, C) or 4-D (N, H, W, C) tensor; flip a bool scalar, rot_k an
    int scalar (Python or tensor). Exposed so gates can apply the identical
    transform to a render (grid gate: render(aug(maps)) == aug(render(maps))).
    """
    x = tf.convert_to_tensor(x)
    flip = tf.cast(flip, tf.bool)              # accepts bool, 0/1, numpy or tensor
    rot_k = tf.cast(rot_k, tf.int32)
    out = tf.cond(flip, lambda: tf.image.flip_left_right(x), lambda: x)
    out = tf.image.rot90(out, k=rot_k)
    # rot90 with a tensor k forgets the static H, W (it may swap them). For a
    # square image every k gives the same shape, so put it back.
    shp = x.shape
    if shp.rank is not None and shp[-3] is not None and shp[-3] == shp[-2]:
        out = tf.ensure_shape(out, shp)
    return out


def crop_and_augment(lr5, hr5, pos_full, off_lr_yx, flip, rot_k, scale, gt_size):
    """One material's crop + augmentation; pure, no randomness inside.

    lr5       (5, h, w, 3) uint8
    hr5       (5, h*scale, w*scale, 3) uint8
    pos_full  (h*scale, w*scale, 2) float32, position_grid_np(h*scale)
    off_lr_yx int (2,) LR offset (y, x); HR and pos are cut at off * scale
    flip      bool scalar; rot_k int scalar in {0,1,2,3} (the rot90 k itself)

    Returns lr (5, gt/scale, gt/scale, 3) float32 in [0,1],
            hr (5, gt, gt, 3) float32 in [0,1], pos (gt, gt, 2) float32.
    """
    scale, gt_size = int(scale), int(gt_size)
    if gt_size % scale:
        raise ValueError("gt_size %d is not a multiple of scale %d" % (gt_size, scale))
    lr_size = gt_size // scale
    lr5 = tf.convert_to_tensor(lr5)
    hr5 = tf.convert_to_tensor(hr5)
    pos_full = tf.convert_to_tensor(pos_full, tf.float32)
    if lr5.dtype != tf.uint8 or hr5.dtype != tf.uint8:
        raise TypeError("crop_and_augment wants uint8 maps (decode_png output), got %s/%s"
                        % (lr5.dtype, hr5.dtype))

    off = tf.cast(off_lr_yx, tf.int32)
    oy, ox = off[0], off[1]
    zero = tf.constant(0, tf.int32)

    # modules/dataset.py:56-57: LR at the offset, HR at offset * scale. The grid
    # is cut exactly like HR.
    lr = tf.slice(lr5, tf.stack([zero, oy, ox, zero]), [-1, lr_size, lr_size, -1])
    hr = tf.slice(hr5, tf.stack([zero, oy * scale, ox * scale, zero]),
                  [-1, gt_size, gt_size, -1])
    pos = tf.slice(pos_full, tf.stack([oy * scale, ox * scale, zero]),
                   [gt_size, gt_size, -1])
    # A -1 size with a runtime begin leaves that dim unknown; restore what is known.
    lr = tf.ensure_shape(lr, [lr5.shape[0], lr_size, lr_size, lr5.shape[-1]])
    hr = tf.ensure_shape(hr, [hr5.shape[0], gt_size, gt_size, hr5.shape[-1]])
    pos = tf.ensure_shape(pos, [gt_size, gt_size, pos_full.shape[-1]])

    # :59-81 - flip then rot, identical for the five maps and the grid.
    lr = augment(lr, flip, rot_k)
    hr = augment(hr, flip, rot_k)
    pos = augment(pos, flip, rot_k)

    # :83-85
    lr = lr / 255
    hr = hr / 255
    return lr, hr, pos


def draw_augmentation(seed, index, limit, using_flip=True, using_rot=True):
    """Stateless (offset (2,) int32, flip bool, rot_k int32) for one element.

    key = [seed, index] split into three keys, one per draw, so the draws are
    independent of each other and of every other element. The distributions are
    upstream's (modules/dataset.py:53-55, 61, 70; see the module docstring).
    `limit` = lr_side - lr_crop + 1. Runs inside tf.data (CPU), where the
    stateless ops' auto-selected algorithm is Philox.
    """
    key = tf.stack([tf.constant(int(seed), tf.int64), tf.cast(index, tf.int64)])
    k_off, k_flip, k_rot = tf.unstack(tf.random.experimental.stateless_split(key, 3))

    offset = tf.random.stateless_uniform(
        [2], seed=k_off, minval=0, maxval=tf.int32.max, dtype=tf.int32) % limit

    flip_case = tf.random.stateless_uniform([], seed=k_flip, minval=0, maxval=2,
                                            dtype=tf.int32)
    flip = tf.equal(flip_case, 0) if using_flip else tf.constant(False)

    rot_case = tf.random.stateless_uniform([], seed=k_rot, minval=0, maxval=4,
                                           dtype=tf.int32)
    rot_k = (rot_case + 1) % 4 if using_rot else tf.constant(0, tf.int32)
    return offset, flip, rot_k


def _read_meta(path):
    meta_path = path + ".meta.json"
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _count_records(path):
    """Example count from the TFRecord framing alone (u64 length, u32 crc, data,
    u32 crc), without reading the payloads. For a record with no meta sidecar."""
    n = 0
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if not head:
                return n
            if len(head) < 12:
                raise ValueError("%s: truncated TFRecord header after %d records" % (path, n))
            f.seek(int(np.frombuffer(head[:8], "<u8")[0]) + 4, os.SEEK_CUR)
            n += 1


def shuffle_buffer_size(path, n_records):
    """min(SHUFFLE_BUFFER, n_records, SHUFFLE_BUFFER_BYTES / mean record bytes), >= 1.

    repeat() comes before shuffle(), so a buffer larger than the record would only
    hold extra copies of it. Falls back to SHUFFLE_BUFFER if the size is unknown."""
    buf = SHUFFLE_BUFFER
    if n_records > 0:
        buf = min(buf, n_records)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        if size > 0:
            buf = min(buf, SHUFFLE_BUFFER_BYTES * n_records // size)
    return max(1, int(buf))


def _check_fixed(fixed, limit):
    """fixed = (off_y, off_x, flip, rot_k) in LR pixels / rot90 k. Python-validated
    so a bad --overfit-fixed fails before tf.data starts."""
    if len(fixed) != 4:
        raise ValueError("fixed must be (off_y, off_x, flip, rot_k), got %r" % (fixed,))
    oy, ox, fl, k = (int(v) for v in fixed)
    if not (0 <= oy < limit and 0 <= ox < limit):
        raise ValueError("fixed LR offset (%d, %d) outside [0, %d]" % (oy, ox, limit - 1))
    if fl not in (0, 1):
        raise ValueError("fixed flip must be 0 or 1, got %r" % (fixed[2],))
    if k not in (0, 1, 2, 3):
        raise ValueError("fixed rot_k must be 0..3, got %r" % (fixed[3],))
    return oy, ox, bool(fl), k


def load_grouped_dataset(cfg, shuffle, seed, start_index=0, fixed=None, overfit_batches=0):
    """Batches of (lr5 (B,5,s,s,3), hr5 (B,5,g,g,3), pos (B,g,g,2)), all float32.

    cfg              the trainer config: batch_size (materials per step), gt_size,
                     scale, train_dataset.{path, using_bin, using_flip, using_rot}
    shuffle          upstream's shuffle, seeded with `seed`; buffer from
                     shuffle_buffer_size() (byte-capped, not 10240)
    seed             int; keys every crop/flip/rot draw (required)
    start_index      global index of the first element = steps_done * batch_size
                     on resume; the stream and its draws continue exactly
    fixed            (off_y, off_x, flip, rot_k) applied to EVERY element instead
                     of the draws (the overfit gate's fixed crop)
    overfit_batches  >0: keep only the first N batches, cached and repeated
    """
    if seed is None:
        raise ValueError("load_grouped_dataset needs an integer seed: the crop/flip/rot "
                         "draws are keyed on it (set `seed:` in the config or --seed)")
    seed = int(seed)
    start_index = int(start_index)
    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    dcfg = cfg["train_dataset"]
    path = dcfg["path"]
    if not dcfg.get("using_bin", True):
        raise ValueError("train_dataset.using_bin is False, but a grouped record only "
                         "holds encoded PNG bytes (modules/dataset.py:6-13's bin branch)")
    using_flip = bool(dcfg.get("using_flip", False))
    using_rot = bool(dcfg.get("using_rot", False))
    batch_size = int(cfg["batch_size"])
    gt_size = int(cfg["gt_size"])
    scale = int(cfg["scale"])

    meta = _read_meta(path)
    hr_side = int(meta.get("hr_size", HR_SIZE))
    lr_side = int(meta.get("lr_size", hr_side // scale))
    if lr_side * scale != hr_side:
        raise ValueError("record sizes hr %d / lr %d do not match scale %d"
                         % (hr_side, lr_side, scale))
    if gt_size % scale or gt_size > hr_side:
        raise ValueError("gt_size %d must be a multiple of scale %d and <= %d"
                         % (gt_size, scale, hr_side))
    lr_crop = gt_size // scale
    limit = lr_side - lr_crop + 1                                     # modules/dataset.py:53
    n_records = int(meta.get("record_count", 0))
    if n_records <= 0 and os.path.isfile(path):
        n_records = _count_records(path)
    if fixed is not None:
        fixed = _check_fixed(fixed, limit)

    pos_full = tf.constant(position_grid_np(hr_side))
    spec = feature_spec()

    def parse(index, record):
        x = tf.io.parse_single_example(record, spec)
        # channels=3 as modules/dataset.py:12-13; grey maps are stored as three
        # equal channels anyway (map_encode.to_rgb8).
        hr5 = tf.stack([tf.image.decode_png(x["image/hr/" + k], channels=3) for k in MAP_KEYS])
        lr5 = tf.stack([tf.image.decode_png(x["image/lr/" + k], channels=3) for k in MAP_KEYS])
        hr5 = tf.ensure_shape(hr5, [N_MAPS, hr_side, hr_side, 3])
        lr5 = tf.ensure_shape(lr5, [N_MAPS, lr_side, lr_side, 3])
        if fixed is None:
            offset, flip, rot_k = draw_augmentation(seed, index, limit, using_flip, using_rot)
        else:
            offset = tf.constant([fixed[0], fixed[1]], tf.int32)
            flip = tf.constant(fixed[2])
            rot_k = tf.constant(fixed[3], tf.int32)
        return crop_and_augment(lr5, hr5, pos_full, offset, flip, rot_k, scale, gt_size)

    ds = tf.data.TFRecordDataset(path)
    ds = ds.repeat()
    buffer = 0
    if shuffle:
        buffer = shuffle_buffer_size(path, n_records)
        ds = ds.shuffle(buffer, seed=seed, reshuffle_each_iteration=True)
        # The shuffled stream is not periodic, so an exact resume has to read
        # every record already consumed (start_index of them) through skip().
        skip = start_index
        if n_records > 0 and start_index > n_records:
            print("[!] shuffled resume: skip() reads %d records (%.1f passes) before the "
                  "first batch" % (start_index, start_index / float(n_records)))
    else:
        # repeat() of a fixed file is periodic: element i is record i % n.
        skip = start_index % n_records if n_records > 0 else start_index
    ds = ds.skip(skip)
    ds = ds.enumerate(start=start_index)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    if overfit_batches > 0:
        ds = ds.take(int(overfit_batches)).cache().repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)

    print("[*] grouped dataset %s: records=%s hr=%d lr=%d crop=%d/%d batch=%d materials "
          "(%d images) shuffle=%s seed=%d start_index=%d skip=%d flip=%s rot=%s%s%s"
          % (path, n_records or "?", hr_side, lr_side, gt_size, lr_crop, batch_size,
             batch_size * N_MAPS, "on (buffer %d)" % buffer if shuffle else "off", seed,
             start_index, skip,
             using_flip, using_rot,
             "" if fixed is None else " fixed=%d,%d,%d,%d" % (fixed[0], fixed[1],
                                                              int(fixed[2]), fixed[3]),
             "" if not overfit_batches else " overfit_batches=%d" % overfit_batches))
    return ds


def parse_names(record_path, limit=0):
    """img_name of every Example (first `limit` if > 0), in record order."""
    spec = {"image/img_name": tf.io.FixedLenFeature([], tf.string)}
    ds = tf.data.TFRecordDataset(record_path)
    if limit and limit > 0:
        ds = ds.take(int(limit))
    ds = ds.map(lambda r: tf.io.parse_single_example(r, spec)["image/img_name"])
    return [n.decode("utf-8") for n in ds.as_numpy_iterator()]
