import tensorflow as tf
import os
import numpy as np
from model import *
from util import *

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/hackathon/05/', 'directory where CIFAR 10 is located')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 50, '')
FLAGS = flags.FLAGS

def main(argv):
    # load text file
    with open("path.txt", "r") as f: 
        path_string = f.read().split(sep='\n')[0]

    save_dir = '/work/' + path_string + '/homework03/logs'

    cifar10_train_data = np.load(FLAGS.data_dir + 'cifar10_train_data.npy')
    cifar10_test_data = np.load(FLAGS.data_dir + 'cifar10_test_data.npy')


if __name__ == "__main__":
    tf.app.run()