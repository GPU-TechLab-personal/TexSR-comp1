"""The MatUp modules the driver reuses verbatim (imported from outside the
submodule, never edited):

    model.FullyConnected, dataLoader.MatUpDataset, train.mlp_forward_pass,
    train.loss_fn (l2,1 + metallic reg), render_utils.batch_render_tn_explicit,
    testing.eval_model, data_utils.output_maps_from_list

This is the module-level header of experiments/common/matup_driver.py:31-40.
"""
from variants.experiments.common import paths

paths.add_matup()
import matup.train as mtrain                      # noqa: E402,F401
from matup import data_utils, light_utils, render_utils, testing   # noqa: E402,F401
from matup.dataLoader import MatUpDataset         # noqa: E402,F401
from matup.model import FullyConnected            # noqa: E402,F401
from matup.renderer import Renderer               # noqa: E402,F401

from variants.experiments.common.render import LIGHT_RADIUS                  # noqa: E402,F401
