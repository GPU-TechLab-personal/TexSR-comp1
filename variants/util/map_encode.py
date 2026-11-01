"""Squeeze any MatSynth map into the 8-bit RGB a DeepTextures network consumes.

One function, shared by both sides of the DeepTextures MatSynth arm:

  * scripts/deeptex/matsynth_make_pairs.py  - builds the TRAINING pairs (dataprep
    container, no TensorFlow)
  * scripts/deeptex/deeptex_compare.py      - feeds the EVALUATION inputs (deeptex
    container)

It lives here, outside both, because deeptex_compare imports TensorFlow at module
level and the dataprep image has none. If the two sides encoded a map differently
the network would be evaluated on a distribution it never trained on, and nothing
would say so - the outputs would just be a little worse.

Only cv2-style arrays go in: whatever cv2.imread(..., IMREAD_UNCHANGED) or
cv2.imdecode returns, i.e. BGR channel order.
"""
import numpy as np


def to_rgb8(arr):
    """BGR/grey, uint8/uint16 -> HxWx3 uint8 RGB.

    * uint16 is rounded, not shifted: a >> 8 biases every value low by up to 1
      LSB. MatSynth height is the 16-bit case.
    * grey is replicated into three equal channels - the upstream loader decodes
      every PNG with channels=3 (modules/dataset.py:12-13).
    * cv2 hands back BGR and the network was trained on RGB (tf.image.decode_png),
      so the reversal is load-bearing. For grey maps it is a no-op.
    """
    if arr.dtype == np.uint16:
        arr = np.rint(arr / 257.0).clip(0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        raise ValueError("unexpected dtype %s" % arr.dtype)

    if arr.ndim == 2:
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.shape[2] == 3:
        rgb = arr[:, :, ::-1]
    elif arr.shape[2] == 4:
        rgb = arr[:, :, 2::-1]
    else:
        raise ValueError("unsupported channel count %d" % arr.shape[2])
    return np.ascontiguousarray(rgb)
