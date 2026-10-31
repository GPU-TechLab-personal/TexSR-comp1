"""Checkpoint restore of the old trainer (deeptex_train.py:674-700), verified.

Replaces train_esrgan.py:65-81 in every arm of ours. Upstream discards
checkpoint.restore()'s status (a structurally mismatched checkpoint restores
partially and silently) and looks for the pretrain checkpoint in
cfg['pretrain_name'] = './checkpoints/', where none can be (defect #2: the
donor builds './checkpoints/' + name). Here:

  * resume: assert_existing_objects_matched();
  * pretrain: cfg['pretrain_dir'] (absolute), fatal (exit 9) when nothing is
    there, and PROVEN by comparing a generator weight before and after.

Shared by the fixed arm (PSNR -> ESRGAN) and the render arm's overfit gate,
which restores the matsynth_all 40k checkpoint the same way.
"""

import hashlib

import numpy as np
import tensorflow as tf

from variants.deeptex.hooks_common.safety import die


def restore(checkpoint, manager, generator, cfg):
    if manager.latest_checkpoint:
        status = checkpoint.restore(manager.latest_checkpoint)
        status.assert_existing_objects_matched()   # upstream discards this
        print("[*] resumed from %s at step %d."
              % (manager.latest_checkpoint, checkpoint.step.numpy()))
    elif cfg.get("pretrain_dir"):
        pretrain_dir = cfg["pretrain_dir"]
        latest = tf.train.latest_checkpoint(pretrain_dir)
        if not latest:
            # Upstream prints and carries on from random init. That is the single
            # biggest unknown in OPEN-ITEMS #6, and it is silent. Not here.
            die(9, "pretrain_dir is set to %s but no checkpoint is there.\n"
                   "        Run --stage psnr first, or set pretrain_dir: null to "
                   "train from scratch on purpose." % pretrain_dir)
        before = generator.trainable_variables[0].numpy().copy()
        checkpoint.restore(latest).expect_partial()   # PSNR ckpt has no D / optimizer_G
        after = generator.trainable_variables[0].numpy()
        if np.array_equal(before, after):
            die(9, "restoring %s did not change the generator's weights.\n"
                   "        The checkpoint resolved but nothing was applied - this "
                   "is exactly the silent no-op OPEN-ITEMS #6 describes." % latest)
        delta = float(np.abs(after - before).max())
        checkpoint.step.assign(0)
        print("[*] pretrain restored from %s (max |dW| = %.4g on '%s')"
              % (latest, delta, generator.trainable_variables[0].name))
    else:
        print("[*] training from scratch (pretrain_dir is null).")


def weights_sha(model):
    """First 16 hex of sha256 over every weight (trainable or not), in
    model.get_weights() order. Equal across two runs = identical initial state;
    the smoke gate compares R0 against R1 with it."""
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()[:16]
