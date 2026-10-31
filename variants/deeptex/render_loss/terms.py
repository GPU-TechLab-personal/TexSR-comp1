"""The render arm's loss terms, built in one place.

The old render_calibrate.py:344-369 rebuilt the trainer's step term by term
(each term with its own tape, so its gradient norm on G can be measured alone).
Both now take the terms from here:

  render_term    the unweighted l2,1 the trainer multiplies by w_render
                 (render_loss/hooks.py extra_loss, gates/render_calibrate.py)
  g_term         one weighted G term exactly as loop.py's step line builds it
                 (train_esrgan.py:95-97) - calibrate only; loop.py keeps the
                 upstream line itself, which L1 checks this against
  remove_offset  calibrate's render_offrm: SR with its per-image, per-channel
                 mean offset to HR taken out (gradient of the detail only)
"""

import tensorflow as tf


def render_term(render_loss, sr5, pos, gt_display, valid):
    return render_loss.loss(sr5, pos, gt_display, valid)


def remove_offset(sr5, hr5):
    return sr5 - tf.stop_gradient(tf.reduce_mean(sr5 - hr5, axis=[2, 3], keepdims=True))


def g_term(term, weights, pixel_loss_fn, fea_loss_fn, gen_loss_fn, discriminator, hr, sr):
    if term == "pixel":
        return weights["w_pixel"] * pixel_loss_fn(hr, sr)
    if term == "feature":
        return weights["w_feature"] * fea_loss_fn(hr, sr)
    if term == "gan":
        hr_output = discriminator(hr, training=True)
        sr_output = discriminator(sr, training=True)
        return weights["w_gan"] * gen_loss_fn(hr_output, sr_output)
    raise ValueError("unknown G term %r" % term)
