# -*- coding: utf-8 -*-
# train.py
# Ben Brittain

import tensorflow as tf
import numpy as np
import reader
import os

from six.moves import cPickle
from model import Model

flags = tf.flags
logging = tf.logging
flags.DEFINE_string("csv_path", None, "csv_path")
FLAGS = flags.FLAGS

def one_hot_to_num(arr):
    ''' don't put non-one hot numbers in here! '''
    return np.nonzero(arr)[0]

class Config(object):
    lr = 0.005
    decay_rate = 0.999
    training_iters = 1250000
    batch_size = 10240
    display_step = 100
    num_layers = 2
    n_input = 3
    n_steps = 5 #timestep

    n_hidden = 600
    n_classes = 150 # potential users
    log_dir = 'keystroke_log'
    save_dir = 'keystroke_models'
    save_frequency = 1000

def main():
    if not FLAGS.csv_path:
        raise ValueError("must set --csv_path")

    villani = reader.read_data_sets(FLAGS.csv_path)

    config = Config()
    model = Model(config)
    with open(os.path.join(config.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(config, f)

    with tf.Session() as sess:
        writer = tf.train.SummaryWriter(config.log_dir, sess.graph)
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        step = 1
        for step in range(config.training_iters):
            new_lr = config.lr * (config.decay_rate ** villani.train.epochs_completed)
            sess.run(tf.assign(model.lr, new_lr))
            batch_xs, batch_ys = villani.train.next_batch(config.batch_size)
            batch_xs = batch_xs.reshape((-1, config.n_steps, config.n_input))
            batch_ys = batch_ys.reshape((-1, config.n_steps, config.n_classes))[:,(config.n_steps-1)]
            _ = sess.run(model.optimizer, feed_dict={
                model.input_data: batch_xs,
                model.targets: batch_ys,
                model.initial_state: np.zeros((batch_xs.shape[0], config.n_hidden))})

            if step % config.display_step == 0:
                # Calculate batch accuracy
                acc, loss, summary = sess.run([model.accuracy, model.cost, model.summary],
                        feed_dict={model.input_data: batch_xs, model.targets: batch_ys,
                            model.initial_state: np.zeros((batch_xs.shape[0], config.n_hidden))})
                print("Index %d, Minibatch Loss= %f, Training Accuracy %f, Learning Rate %f"%((step, loss, acc, new_lr)))
                writer.add_summary(summary)
                writer.flush()

            if step % config.save_frequency == 0:
                checkpoint_path = os.path.join(config.save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = step)

        print("Optimization Finished!")

if __name__ == "__main__":
    main()
