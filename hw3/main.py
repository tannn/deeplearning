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

    x = tf.placeholder(shape=[None, 16641], dtype=tf.float32, name='input_placeholder')

    output_dense = dense_block(flat_english, language="english")
    output = tf.identity(output_dense_english, name='output2')

    optimizer, cross_entropy, train_op = optimizer_block("english", output_dense, y, 0.001)
    sum_cross_entropy = tf.reduce_mean(cross_entropy_english)

    #peak signal to noise ratio
    mse = tf.reduce_mean(tf.squared_difference(image_target,output))
    PSNR = tf.constant(255**2,dtype=tf.float32)
    PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR) * 20


    with tf.Session() as session:


        ce_vals = []
        for i in range(test_num_examples // batch_size):
            batch_xs = test_images[i*batch_size:(i+1)*batch_size, :]
            test_ce = session.run([sum_cross_entropy], {x: batch_xs})
            ce_vals.append(test_ce)
        avg_test_ce = sum(ce_vals) / len(ce_vals)
        print('TEST CROSS ENTROPY: ' + str(avg_test_ce))





if __name__ == "__main__":
    tf.app.run()