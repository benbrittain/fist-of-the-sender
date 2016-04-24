# -*- coding: utf-8 -*-
# train.py
# Ben Brittain

import tensorflow as tf
import numpy as np
import reader
from tensorflow.models.rnn import rnn, rnn_cell

flags = tf.flags
logging = tf.logging
flags.DEFINE_string("csv_path", None, "csv_path")
FLAGS = flags.FLAGS


def one_hot_to_num(arr):
    ''' don't put non-one hot numbers in here! '''
    return np.nonzero(arr)[0]

class Config(object):
    lr = 0.001
    training_iters = 12500
    batch_size = 360
    display_step = 10
    num_layers = 2
    n_input = 3
    n_steps = 20 #timestep

    n_hidden = 128
    n_classes = 256 # potential users

class Model(object):
    def __init__(self, config):
        # Tf Graph input
        self.input_data = tf.placeholder(tf.float32, [None, config.n_steps, config.n_input], name='input')

        # Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
        self.initial_state = tf.placeholder(tf.float32, [None, 2*config.n_hidden], name='state')
        self.targets = tf.placeholder(tf.float32, [None, config.n_classes], name='target')

        _X = tf.transpose(self.input_data, [1, 0, 2])  # permute n_steps and batch_size
        _X = tf.reshape(_X, [-1, config.n_input]) # (n_steps*batch_size, n_input)

        # Define a lstm cell with tensorflow
        cell = lstm_cell = rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)
#        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

#        lstm_cell = rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=0.5)

        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(0, config.n_steps, _X) # n_steps * (batch_size, n_hidden)

        # LSTM cell output
        outputs, states = rnn.rnn(cell, _X, initial_state=self.initial_state)
        #output = tf.reshape(tf.concat(1, outputs), [-1, config.n_hidden])

        with tf.variable_scope('output'):
            softmax_w = tf.get_variable("softmax_w", [config.n_hidden, config.n_classes])
            softmax_b = tf.get_variable("softmax_b", [config.n_classes])
        self.logits = logits = tf.matmul(outputs[-1], softmax_w) + softmax_b

        # Loss and optimizer
        self.cost = cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.targets))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=config.lr).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(self.targets,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def main():
    if not FLAGS.csv_path:
        raise ValueError("must set --csv_path")

    villani = reader.read_data_sets(FLAGS.csv_path)

    config = Config()
    model = Model(config)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        step = 1
        for step in range(config.training_iters):
            batch_xs, batch_ys = villani.train.next_batch(config.batch_size)
            batch_xs = batch_xs.reshape((-1, config.n_steps, config.n_input))
            batch_ys = batch_ys.reshape((-1, config.n_steps, config.n_classes))[:,7]
            _ = sess.run(model.optimizer, feed_dict={
                model.input_data: batch_xs,
                model.targets: batch_ys,
                model.initial_state: np.zeros((batch_xs.shape[0], 2*config.n_hidden))})

            if step % config.display_step == 0:
                # Calculate batch accuracy
                acc, loss = sess.run([model.accuracy, model.cost],
                        feed_dict={model.input_data: batch_xs, model.targets: batch_ys,
                            model.initial_state: np.zeros((batch_xs.shape[0], 2*config.n_hidden))})
                print("Index %d, Minibatch Loss= %f, Training Accuracy %f"%((villani.train.index_in_epoch, loss, acc)))
        print("Optimization Finished!")

#        test_start = 0
#        test_end = 120
#        test_len = test_end - test_start
#        test_data = villani.test.keystrokes[test_start:test_end].reshape((-1, config.n_steps, config.n_input))
#        test_label = villani.test.labels[test_start:test_end].reshape((-1, config.n_steps, config.n_classes))[:,7]
#        print("Testing Accuracy:", sess.run(model.accuracy, feed_dict={model.input_data: test_data,
#                                                                 model.targets: test_label,
#                                                                 model.initial_state:
#                                                                 np.zeros((test_data.shape[0], 2*config.n_hidden))}))
#
#        test_start = 6000
#        test_end = 6030
#        test_len = test_end - test_start
#        test_data = villani.test.keystrokes[test_start:test_end].reshape((-1, config.n_steps, config.n_input))
#        test_label = villani.test.labels[test_start:test_end].reshape((-1, config.n_steps, config.n_classes))[:,7]
#        print("Testing Accuracy:", sess.run(model.accuracy, feed_dict={model.input_data: test_data,
#                                                                 model.targets: test_label,
#                                                                 model.initial_state:
#                                                                 np.zeros((test_data.shape[0], 2*config.n_hidden))}))

if __name__ == "__main__":
    main()
