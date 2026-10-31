"""The 'init' variant (change (j)): MatUp's starting point, no optimisation."""
from variants.experiments.matup_driver.upstream import MatUpDataset, Renderer, data_utils


def init_maps(args, device):
    """MatUp's starting point (bilinear x4 of the LR maps) over the same valid
    region as its output - the 'did the optimisation help at all' reference."""
    # (j) cf. main_optim.output_initial_maps (main_optim.py:23-33), which writes
    # the same region as PNGs; here it is returned, renorm-clamped like the output
    Renderer.init_renderer(args, device)
    dataset = MatUpDataset(args, device)
    b = (args.tile_width - 1) // 2
    v = dataset.get_svbrdf_init()[b:-b, b:-b].to(device)
    pos = dataset.get_position3D()[b:-b, b:-b].to(device)
    maps = data_utils.get_renorm_clamped_maps(data_utils.maps_from_data(v, pos))
    return maps, dataset
