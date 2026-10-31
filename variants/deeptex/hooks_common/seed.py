"""Seeding of the old trainer (deeptex_train.py:563-577). Only when a seed is given.

Called before check_gpu's random matmul and the model build. tf.random.set_seed
alone does NOT fix the init under Keras 2.10: VarianceScaling (every G/D conv,
modules/models.py:11-15) and glorot_uniform (D's Dense) take their stateless
seed from backend._SEED_GENERATOR, else from Python's unseeded global `random`.
set_random_seed seeds random, np.random, tf.random AND that generator: identical
G/D init in every run with this seed. Upstream never seeds.
"""

import tensorflow as tf

from variants.deeptex.hooks_common.safety import die


def apply_seed(seed):
    """None = unseeded (the paper's behaviour). Returns the int seed or None."""
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            die(8, "seed = %r is not an integer" % (seed,))
        if not 0 <= seed < 2 ** 32:
            die(8, "seed = %d is outside [0, 2**32), which np.random.seed rejects" % seed)
        tf.keras.utils.set_random_seed(seed)
        print("[*] tf.keras.utils.set_random_seed(%d)" % seed)
    return seed
