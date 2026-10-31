from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf

from modules.models import RRDB_Model
from modules.lr_scheduler import MultiStepLR
from modules.losses import PixelLoss
from modules.utils import (load_yaml, load_dataset, ProgressBar,
                           set_memory_growth)
from variants.deeptex.paper_loop.hooks import PsnrHooks  # hooks: every default is the line it replaced


flags.DEFINE_string('cfg_path', './configs/psnr.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')


def main(_, hooks=None):
    hooks = hooks or PsnrHooks()
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    hooks.on_start(logger)

    cfg = hooks.load_cfg(FLAGS.cfg_path)
    hooks.pre_build(cfg)

    # define network
    model = RRDB_Model(cfg['input_size'], cfg['ch_size'], cfg['network_G'])
    hooks.summary(model)

    # load dataset
    train_dataset = hooks.build_dataset(cfg)

    # define optimizer
    learning_rate = MultiStepLR(cfg[hooks.lr_key], cfg['lr_steps'], cfg['lr_rate'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=cfg['adam_beta1_G'],
                                         beta_2=cfg['adam_beta2_G'])

    # define losses function
    pixel_loss_fn = PixelLoss(criterion=cfg['pixel_criterion'])

    # load checkpoint
    checkpoint_dir = hooks.checkpoint_dir(cfg)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer=optimizer,
                                     model=model)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=hooks.max_to_keep)
    hooks.restore(checkpoint, manager, model, cfg)

    # define training step function
    @hooks.step_function
    def train_step(lr, hr, *extras):  # hooks: extras = what hooks.batches yields beyond (lr, hr)
        lr, hr, extras = hooks.pre_tape(lr, hr, extras)
        with tf.GradientTape() as tape:
            sr = model(lr, training=True)

            losses = {}
            losses['reg'] = tf.reduce_sum(model.losses)
            losses['pixel'] = cfg['w_pixel'] * pixel_loss_fn(hr, sr)
            total_loss = tf.add_n([l for l in losses.values()])

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return (total_loss, losses) + hooks.step_outputs(sr, hr, extras)

    # training loop
    summary_writer = tf.summary.create_file_writer(
        hooks.log_dir(cfg))
    prog_bar = ProgressBar(cfg['niter'], checkpoint.step.numpy())
    remain_steps = max(cfg['niter'] - checkpoint.step.numpy(), 0)
    train_dataset = hooks.before_loop(train_dataset, checkpoint, cfg)

    for lr, hr, *extras in hooks.batches(train_dataset, remain_steps):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()

        total_loss, losses, *outs = hooks.call_step(train_step, steps, lr, hr, *extras)

        hooks.after_step(prog_bar, steps, total_loss, losses, outs, optimizer)

        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar(
                    'loss/total_loss', total_loss, step=steps)
                for k, l in losses.items():
                    tf.summary.scalar('loss/{}'.format(k), l, step=steps)
                tf.summary.scalar(
                    'learning_rate', optimizer.lr(steps), step=steps)
                hooks.write_summaries(steps, outs)

        if steps % cfg['save_steps'] == 0:
            manager.save()
            hooks.saved(manager)

    hooks.on_end(checkpoint, manager, remain_steps)


if __name__ == '__main__':
    app.run(main)
