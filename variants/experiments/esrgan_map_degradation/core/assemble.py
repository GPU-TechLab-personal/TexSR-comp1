"""Which arm builds which part, read off the config.

The config keys are the old ones (gate_configs checks every arm differs from
its reference in the named factor only), so the arm is never named in a config;
it is the set of factors these predicates find. The predicates are the old
if/else branches of experiments/exp2_esrgan/{train,model,losses}.py, moved here
one to one.

part             config                                 built by                         arms
generator        model.input == concat                  exp2 arms/concat.ConcatSR        2-4, 3-0..3-2
                 model.arch == shared                   exp2 arms/base.SharedSR          2-0, 2-1, 2-3
                 model.arch == separate                 exp2 arms/separate.SeparateSR    2-2
init             init.mode == pretrained                exp2 arms/pretrained.load        2-1, 3-0..3-2
                 init.mode == scratch                   nothing (base)                   2-0, 2-2..2-4
discriminators   disc.mode per_map, GAN maps, arch sep. exp2 arms/separate.PerMapD       2-2 (5 Ds)
                 disc.mode per_map, GAN maps            exp2 arms/base.SharedD           2-0, 2-1, 2-3, 2-4, 3-0, 3-1
                 disc.mode joint                        exp3 arms/joint_d.JointD         3-2
                 otherwise                              exp2 arms/base.Discriminators    (no D)
pixel terms      always                                 base.L1Term
                 any per_map spec has `angular`         mapaware_loss.AngularTerm        2-3
map terms        loss.extra_l1 non-empty                exp3 render_loss.ExtraL1Term     3-1, 3-2
                 always                                 base.VGGTerm, base.PerMapGanTerm
post terms       loss.joint_gan > 0                     exp3 joint_d.JointGanTerm        3-2
                 loss.render.weight > 0                 exp3 render_loss.RenderTerm      3-1, 3-2

Construction order is the old trainer's (G, init, D, VGG, RenderLoss): every
module that draws initial weights draws them in the same order from the same
seed, so a fresh run starts from the same weights as before.
"""
import collections

from variants.experiments.esrgan_map_degradation.arms import base, concat, mapaware_loss, pretrained, separate
from variants.experiments.esrgan_map_degradation.core.model import MAPS, VGGPerceptual, VGGStyleDiscriminator
from variants.experiments.esrgan_render_loss.arms import joint_d, render_loss

Parts = collections.namedtuple("Parts", "G init_report discs crit gan_maps dmode")


def generator(arch="shared", input="independent", num_block=23, num_feat=64, num_grow_ch=32):
    """MultiMapSR(arch, input, ...) of the old model.py, by arm."""
    if input not in ("independent", "concat") or arch not in ("shared", "separate"):
        raise ValueError((arch, input))
    if input == "concat" and arch == "separate":
        raise ValueError("concat input means one network over all maps; arch must be shared")
    if input == "concat":
        return concat.ConcatSR(num_block, num_feat, num_grow_ch)
    elif arch == "shared":
        return base.SharedSR(num_block, num_feat, num_grow_ch)
    return separate.SeparateSR(num_block, num_feat, num_grow_ch)


def discriminators(mode, gan_maps, separate_g, make):
    """Discriminators(mode, gan_maps, separate, make) of the old losses.py, by arm."""
    gan_maps = list(gan_maps)
    if mode == "per_map" and gan_maps:
        if separate_g:
            return separate.PerMapD(mode, gan_maps, separate_g, make)
        return base.SharedD(mode, gan_maps, separate_g, make)
    elif mode == "joint":
        return joint_d.JointD(mode, gan_maps, separate_g, make)
    return base.Discriminators(mode, gan_maps, separate_g)


def criterion(loss_cfg, vgg=None, render=None):
    """Criterion(cfg, device, vgg) + `crit.render = render` of the old losses.py, by arm."""
    per_map = loss_cfg["per_map"]
    extra_l1 = loss_cfg.get("extra_l1", {}) or {}
    joint_gan = float(loss_cfg.get("joint_gan", 0.0))

    pixel = [base.L1Term()]
    if mapaware_loss.applies(per_map):
        pixel.append(mapaware_loss.AngularTerm())
    map_terms = []
    if extra_l1:
        map_terms.append(render_loss.ExtraL1Term(extra_l1))
    map_terms += [base.VGGTerm(per_map, vgg), base.PerMapGanTerm(per_map)]
    post = []
    if joint_gan > 0:
        post.append(joint_d.JointGanTerm(joint_gan))
    if render is not None:
        post.append(render_loss.RenderTerm(render))
    return base.Criterion(per_map, pixel, map_terms, post)


def build(cfg, dev, pretrained_dir):
    """The '---- model' block of experiments/exp2_esrgan/train.py:131-149."""
    mc = cfg["model"]
    G = generator(mc["arch"], mc["input"], mc.get("num_block", 23), mc.get("num_feat", 64),
                  mc.get("num_grow_ch", 32)).to(dev)
    init_report = {"mode": cfg["init"]["mode"]}
    if cfg["init"]["mode"] == "pretrained":
        init_report.update(pretrained.load(G, cfg["init"]["weights"], pretrained_dir))
    elif cfg["init"]["mode"] != "scratch":
        raise ValueError(cfg["init"]["mode"])

    per_map = cfg["loss"]["per_map"]
    gan_maps = [m for m in MAPS if per_map.get(m, {}).get("gan", 0) > 0]
    dmode = cfg["disc"]["mode"]
    discs = discriminators(dmode, gan_maps, separate.owns_discriminators(mc),
                           lambda c: VGGStyleDiscriminator(c, cfg["disc"].get("num_feat", 64))).to(dev)
    need_vgg = any(v.get("vgg", 0) > 0 for v in per_map.values())
    vgg = VGGPerceptual().to(dev) if need_vgg else None
    render = None
    if render_loss.applies(cfg["loss"]):
        render = render_loss.RenderLoss(cfg["loss"]["render"], dev, cfg["data"]["hr_patch"])
    crit = criterion(cfg["loss"], vgg, render)
    return Parts(G, init_report, discs, crit, gan_maps, dmode)
