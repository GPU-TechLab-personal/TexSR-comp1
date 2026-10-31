"""MatUp's command line as a Namespace (change (i), see README.md)."""
import argparse


def matup_args(lr_dir, render_dir, up_dir, out_dir, **over):
    """main.py's argparse defaults, with --ogl_normals (MatSynth is OpenGL)."""
    # (i) every value is main.py:12-53's default except opengl_normals=True:
    #     MatUp defaults to DirectX normals, MatSynth ships OpenGL ones.
    a = dict(upscaler="swinir", input_mat=lr_dir.rstrip("/") + "/",
             render_dir=render_dir.rstrip("/") + "/", upsamplings_dir=up_dir.rstrip("/") + "/",
             output_folder=out_dir.rstrip("/") + "/", output_init=False,
             tile_width=17, batch_size=128, nb_epochs=10, learning_rate=1e-4, alpha_reg=1e-1,
             save_interval=999, hid_layer_width=128, hid_layer_size=4,
             non_linearity="LeakyReLU", crop_size=128, AA_upscale=2, nb_lights=100,
             disp_val=0.01, light_falloff=False, opengl_normals=True, temperature=0.5,
             material_size=128, up_batch_size=10)
    a.update(over)
    return argparse.Namespace(**a)
