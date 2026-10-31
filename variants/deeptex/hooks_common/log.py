"""Newline-terminated log lines of the old trainer (deeptex_train.py:908-928).

Replaces train_esrgan.py:125-128's ProgressBar, which writes \\r with no newline
so a piped log becomes one enormous line. The orchestrators parse these prefixes
and nothing else, so their format is a contract (variants/README.md rule 4):

  [metrics] step= niter= loss_G= loss_D= lr_G= steps_per_sec= gpu_mem_mb= eta_h=
  [terms] step= reg= pixel= feature= gan= [render=] [mask_frac= oor_* sr_nz_guard=]

Not log.py-as-in-stdlib: named `log` so nothing shadows `logging`.
"""

import time

import tensorflow as tf

# [terms] names for the step's diagnostics dict, in print order.
#   mask_frac   GT texels dropped from the render mean (invalid GT normal)
#   oor_*       SR values outside [0,1] before the decode clips them (no render
#               gradient there); see render_tf.RenderLoss.diagnostics
#   sr_nz_guard SR texels whose normal the n_z >= nz_min guard moved
TERM_DIAG = (("mask_frac", "mask_frac"), ("oor_basecolor", "oor_bc"),
             ("oor_roughness", "oor_rough"), ("oor_metallic", "oor_metal"),
             ("oor_height", "oor_height"), ("sr_nz_guard", "sr_nz_guard"))


class MetricsWindow(object):
    """steps_per_sec over the steps since the last [metrics] line."""

    def __init__(self):
        self.t_win = time.time()
        self.win_steps = 0

    def tick(self):
        self.win_steps += 1

    def line(self, steps, niter, g, d, optimizer_G):
        dt = time.time() - self.t_win
        sps = self.win_steps / dt if dt > 0 else 0.0
        self.t_win, self.win_steps = time.time(), 0
        mem = tf.config.experimental.get_memory_info("GPU:0")["peak"] // (1 << 20) \
            if tf.config.list_physical_devices("GPU") else 0
        eta = (niter - steps) / sps / 3600.0 if sps > 0 else float("nan")
        # Newline-terminated and machine-readable. run_deeptex.ps1 parses THIS,
        # never the \r progress bar.
        print("[metrics] step=%d niter=%d loss_G=%.4f loss_D=%.4f lr_G=%.2e "
              "steps_per_sec=%.2f gpu_mem_mb=%d eta_h=%.2f"
              % (steps, niter, g, d, float(optimizer_G.lr(steps).numpy()),
                 sps, mem, eta), flush=True)


def terms_line(steps, losses_G, diag):
    """The weighted G terms one by one ([metrics] only has their sum), then the
    render diagnostics when a RenderLoss exists."""
    parts = ["%s=%.5g" % (k, float(losses_G[k].numpy()))
             for k in ("reg", "pixel", "feature", "gan", "render") if k in losses_G]
    parts += ["%s=%.5g" % (short, float(diag[k].numpy()))
              for k, short in TERM_DIAG if k in diag]
    print("[terms] step=%d %s" % (steps, " ".join(parts)), flush=True)
