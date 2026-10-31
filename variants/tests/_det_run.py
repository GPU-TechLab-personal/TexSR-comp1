#!/usr/bin/env python3
"""Run a training script deterministically without editing it.

    python _det_run.py [--paper-seed N] SCRIPT [args...]

Turns on tf.config.experimental.enable_op_determinism() and then execs SCRIPT
with runpy as __main__, so the old trainer, the new main.py and the submodule's
own train_esrgan.py all run unmodified under the same op-level determinism.
--paper-seed calls tf.keras.utils.set_random_seed first - only for
train_esrgan.py (L1a), which never seeds itself; our trainers take --seed.
"""

import os
import runpy
import sys


def main():
    argv = sys.argv[1:]
    seed = None
    if argv[:1] == ["--paper-seed"]:
        seed, argv = int(argv[1]), argv[2:]
    script = argv[0]
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    # Op determinism does not cover upstream's loader: modules/dataset.py:99-101
    # maps with num_parallel_calls=AUTOTUNE and draws its crops with STATEFUL
    # tf.random.uniform, so which element gets which draw depends on thread
    # timing (upstream vs itself diverges at step 4 without this). Run every
    # map sequentially - in both the old and the new trainer alike.
    _map = tf.data.Dataset.map

    def sequential_map(self, map_func, num_parallel_calls=None, deterministic=None, name=None):
        return _map(self, map_func, name=name)
    tf.data.Dataset.map = sequential_map
    # And the step: even with op determinism on, multi-threaded CPU kernels left
    # upstream-vs-itself checkpoints differing by ~1e-8..1e-6 after 5 steps.
    # One thread each makes a run reproducible to the bit.
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    sys.argv = argv
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
