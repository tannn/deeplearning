import tensorflow as tf
import os
import numpy as np
from model import *
from util import *

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/02/EMODB-German/', 'directory where EMODB-German is located')
flags.DEFINE_string('username', 'tmarino', '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 100, '')
FLAGS = flags.FLAGS

def main(argv):
    # load data for first fold

    save_dir = '/work/cse496dl/' + FLAGS.username + '/homework02/logs'

    train_images_1 = np.load(FLAGS.data_dir + 'train_x_1.npy')
    train_labels_1 = np.load(FLAGS.data_dir + 'train_y_1.npy')

    test_images_1 = np.load(FLAGS.data_dir + 'test_x_1.npy')
    test_labels_1 = np.load(FLAGS.data_dir + 'test_y_1.npy')

    train_images_2 = np.load(FLAGS.data_dir + 'train_x_2.npy')
    train_labels_2 = np.load(FLAGS.data_dir + 'train_y_2.npy')

    test_images_2 = np.load(FLAGS.data_dir + 'test_x_2.npy')
    test_labels_2 = np.load(FLAGS.data_dir + 'test_y_2.npy')

    train_images_3 = np.load(FLAGS.data_dir + 'train_x_3.npy')
    train_labels_3 = np.load(FLAGS.data_dir + 'train_y_3.npy')

    test_images_3 = np.load(FLAGS.data_dir + 'test_x_3.npy')
    test_labels_3 = np.load(FLAGS.data_dir + 'test_y_3.npy')

    train_images_4 = np.load(FLAGS.data_dir + 'train_x_4.npy')
    train_labels_4 = np.load(FLAGS.data_dir + 'train_y_4.npy')

    test_images_4 = np.load(FLAGS.data_dir + 'test_x_4.npy')
    test_labels_4 = np.load(FLAGS.data_dir + 'test_y_4.npy')

    valid_images_1, test_images_1, valid_labels_1, test_labels_1 = split_data(test_images_1, test_labels_1, 0.1)
    valid_images_2, test_images_2, valid_labels_2, test_labels_2 = split_data(test_images_2, test_labels_2, 0.1)
    valid_images_3, test_images_3, valid_labels_3, test_labels_3 = split_data(test_images_3, test_labels_3, 0.1)
    valid_images_4, test_images_4, valid_labels_4, test_labels_4 = split_data(test_images_4, test_labels_4, 0.1)

    train_images = [train_images_1, train_images_2, train_images_3, train_images_4]
    train_labels = [train_labels_1, train_labels_2, train_labels_3, train_labels_4]
    test_images = [test_images_1, test_images_2, test_images_3, test_images_4]
    test_labels = [test_labels_1, test_labels_2, test_labels_3, test_labels_4]
    valid_images = [valid_images_1, valid_images_2, valid_images_3, valid_images_4]
    valid_labels = [valid_labels_1, valid_labels_2, valid_labels_3, valid_labels_4]

    x = tf.placeholder(shape=[None, 129, 129, 1], dtype=tf.float32, name='input_placeholder')
    conv_x = my_conv_block(x, [16, 32, 64])
    output = output_block(conv_x)

	







if __name__ == "__main__":
    tf.app.run()