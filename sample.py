# -*- coding: utf-8 -*-
# sample.py
# Ben Brittain

import argparse
import tensorflow as tf
from six.moves import cPickle
from train import Config
from model import Model
import os

def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.sample(sess, args.csv, args.s, args.n))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
            help='model directory to store checkpointed models')
    parser.add_argument('--csv', type=str, default='data/keystrokes.csv',
            help='dataset to sample from')
    parser.add_argument('-s', type=int, default=0,
            help='start of sampling')
    parser.add_argument('-n', type=int, default=30,
            help='number of keystrokes to sample')
    args = parser.parse_args()
    sample(args)

if __name__ == "__main__":
    main()
