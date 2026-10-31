from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf

from modules.models import RRDB_Model, DiscriminatorVGG128
from modules.lr_scheduler import MultiStepLR
from modules.losses import (PixelLoss, ContentLoss, DiscriminatorLoss,
                            GeneratorLoss)
from modules.utils import (load_yaml, load_dataset, ProgressBar,
                           set_memory_growth)
from variants.deeptex.paper_loop.hooks import Hooks  # hooks: every default is the line it replaced


flags.DEFINE_string('cfg_path', './configs/esrgan.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')


def main(_, hooks=None):
    hooks = hooks or Hooks()
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    hooks.on_start(logger)

    cfg = hooks.load_cfg(FLAGS.cfg_path)
    hooks.pre_build(cfg)

    # define network
    generator = RRDB_Model(cfg['input_size'], cfg['ch_size'], cfg['network_G'])
    hooks.summary(generator)
    discriminator = DiscriminatorVGG128(cfg['gt_size'], cfg['ch_size'])
    hooks.summary(discriminator)

    # load dataset
    train_dataset = hooks.build_dataset(cfg)

    # define optimizer
    learning_rate_G = MultiStepLR(cfg['lr_G'], cfg['lr_steps'], cfg['lr_rate'])
    learning_rate_D = MultiStepLR(cfg['lr_D'], cfg['lr_steps'], cfg['lr_rate'])
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=learning_rate_G,
                                           beta_1=cfg['adam_beta1_G'],
                                           beta_2=cfg['adam_beta2_G'])
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=learning_rate_D,
                                           beta_1=cfg['adam_beta1_D'],
                                           beta_2=cfg['adam_beta2_D'])

    # define losses function
    pixel_loss_fn = PixelLoss(criterion=cfg['pixel_criterion'])
    fea_loss_fn = ContentLoss(criterion=cfg['feature_criterion'])
    gen_loss_fn, dis_loss_fn = hooks.build_losses(cfg)

    # load checkpoint
    checkpoint_dir = hooks.checkpoint_dir(cfg)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer_G=optimizer_G,
                                     optimizer_D=optimizer_D,
                                     model=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=hooks.max_to_keep)
    hooks.restore(checkpoint, manager, generator, cfg)

    # define training step function
    @hooks.step_function
    def train_step(lr, hr, *extras):  # hooks: extras = what hooks.batches yields beyond (lr, hr)
        lr, hr, extras = hooks.pre_tape(lr, hr, extras)
        with tf.GradientTape(persistent=True) as tape:
            sr = generator(lr, training=True)
            hr_output = discriminator(hr, training=True)
            sr_output = discriminator(sr, training=True)

            losses_G = {}
            losses_D = {}
            losses_G['reg'] = tf.reduce_sum(generator.losses)
            losses_D['reg'] = tf.reduce_sum(discriminator.losses)
            losses_G['pixel'] = cfg['w_pixel'] * pixel_loss_fn(hr, sr)
            losses_G['feature'] = cfg['w_feature'] * fea_loss_fn(hr, sr)
            losses_G['gan'] = cfg['w_gan'] * gen_loss_fn(hr_output, sr_output)
            losses_D['gan'] = dis_loss_fn(hr_output, sr_output)
            losses_G.update(hooks.extra_loss(sr, hr, extras))
            total_loss_G = tf.add_n([l for l in losses_G.values()])
            total_loss_D = tf.add_n([l for l in losses_D.values()])

        grads_G = tape.gradient(
            total_loss_G, generator.trainable_variables)
        grads_D = tape.gradient(
            total_loss_D, discriminator.trainable_variables)
        with tf.control_dependencies(hooks.pre_apply(grads_G, grads_D)):
            optimizer_G.apply_gradients(
                zip(grads_G, generator.trainable_variables))
            optimizer_D.apply_gradients(
                zip(grads_D, discriminator.trainable_variables))

        return (total_loss_G, total_loss_D, losses_G, losses_D) + hooks.step_outputs(sr, hr, extras)

    # training loop
    summary_writer = tf.summary.create_file_writer(
        hooks.log_dir(cfg))
    prog_bar = ProgressBar(cfg['niter'], checkpoint.step.numpy())
    remain_steps = max(cfg['niter'] - checkpoint.step.numpy(), 0)
    train_dataset = hooks.before_loop(train_dataset, checkpoint, cfg)

    for lr, hr, *extras in hooks.batches(train_dataset, remain_steps):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()

        total_loss_G, total_loss_D, losses_G, losses_D, *outs = hooks.call_step(train_step, steps, lr, hr, *extras)

        hooks.after_step(prog_bar, steps, total_loss_G, total_loss_D, losses_G, losses_D, outs, optimizer_G, optimizer_D)

        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar(
                    'loss_G/total_loss', total_loss_G, step=steps)
                tf.summary.scalar(
                    'loss_D/total_loss', total_loss_D, step=steps)
                for k, l in losses_G.items():
                    tf.summary.scalar('loss_G/{}'.format(k), l, step=steps)
                for k, l in losses_D.items():
                    tf.summary.scalar('loss_D/{}'.format(k), l, step=steps)

                tf.summary.scalar(
                    'learning_rate_G', optimizer_G.lr(steps), step=steps)
                tf.summary.scalar(
                    'learning_rate_D', optimizer_D.lr(steps), step=steps)
                hooks.write_summaries(steps, outs)

        if steps % cfg['save_steps'] == 0:
            manager.save()
            hooks.saved(manager)

    hooks.on_end(checkpoint, manager, remain_steps)


if __name__ == '__main__':
    app.run(main)
