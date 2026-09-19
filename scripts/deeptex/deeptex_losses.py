#!/usr/bin/env python3
"""Numerically stable RaGAN losses, selectable against the upstream ones.

external/DeepTextures/modules/losses.py:56,80 builds
BinaryCrossentropy(from_logits=False) and then feeds it values it has ALREADY
passed through tf.sigmoid (:61-62, :85-86). Keras clips y_pred to [1e-7, 1-1e-7]
before taking the log, so this does not produce NaN - the failure is quieter than
that:

    once D saturates, sigmoid(hr - reduce_mean(sr)) reaches exactly 1.0 or 0.0 in
    float32 (|x| > ~17), the clip pins the loss near 16.1, and the gradient through
    the sigmoid collapses to ~0. D stops learning, G's w_gan term goes flat, and
    train_esrgan.py:23-25 has disabled the logger so nothing says a word.

Passing the raw logits to from_logits=True is the same function mathematically and
computes log(sigmoid(x)) via the stable softplus form, so the gradient survives.

With no PSNR pre-training - which upstream's pretrain lookup makes structurally
impossible, see train_esrgan.py:71 - D saturating early is the EXPECTED failure
mode, not a rare one. That is why this is worth fixing rather than noting.

--losses upstream keeps the original behaviour so the difference stays measurable
rather than asserted. PixelLoss and ContentLoss are imported unmodified from the
submodule; only the two adversarial terms are touched here.
"""

import tensorflow as tf


def DiscriminatorLoss(gan_type="ragan"):
    """Mirror of modules/losses.py:54-75 with from_logits=True and no manual sigmoid."""
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss_ragan(hr, sr):
        return 0.5 * (
            cross_entropy(tf.ones_like(hr), hr - tf.reduce_mean(sr)) +
            cross_entropy(tf.zeros_like(sr), sr - tf.reduce_mean(hr)))

    def discriminator_loss(hr, sr):
        real_loss = cross_entropy(tf.ones_like(hr), hr)
        fake_loss = cross_entropy(tf.zeros_like(sr), sr)
        return real_loss + fake_loss

    if gan_type == "ragan":
        return discriminator_loss_ragan
    if gan_type == "gan":
        return discriminator_loss
    raise NotImplementedError(
        "Discriminator loss type {} is not recognized.".format(gan_type))


def GeneratorLoss(gan_type="ragan"):
    """Mirror of modules/losses.py:78-97 with from_logits=True and no manual sigmoid."""
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def generator_loss_ragan(hr, sr):
        return 0.5 * (
            cross_entropy(tf.ones_like(sr), sr - tf.reduce_mean(hr)) +
            cross_entropy(tf.zeros_like(hr), hr - tf.reduce_mean(sr)))

    def generator_loss(hr, sr):
        return cross_entropy(tf.ones_like(sr), sr)

    if gan_type == "ragan":
        return generator_loss_ragan
    if gan_type == "gan":
        return generator_loss
    raise NotImplementedError(
        "Generator loss type {} is not recognized.".format(gan_type))
