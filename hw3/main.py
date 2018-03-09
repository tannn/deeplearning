import tensorflow as tf
import os
import numpy as np
from model import *
from util import *

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/hackathon/05/', 'directory where CIFAR 10 is located')
flags.DEFINE_integer('max_epoch_num', 50, '')
FLAGS = flags.FLAGS

def main(argv):
    batch_size = 32
    # load text file
    with open("path.txt", "r") as f: 
        path_string = f.read().split(sep='\n')[0]

    save_dir = '/work/' + path_string + '/homework03/logs'

    train_data = np.load(FLAGS.data_dir + 'cifar10_train_data.npy')
    test_data = np.load(FLAGS.data_dir + 'cifar10_test_data.npy')

    train_num_examples = train_data.shape[0]
    test_num_examples = test_data.shape[0]

    print(train_data.shape)

    #TODO: Reshape incoming data to be square

    x = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='encoder_input')

    encoder_16 = downscale_block(x)
    encoder_8 = downscale_block(encoder_16)
    flat = flatten(enecoder_8)
    code = tf.layers.dense(flat, 40, activation=tf.nn.relu, name='encoder_output')
    #TODO: create the code as a flatten -> dense 

    decoder_input = tf.identity(code, name="decoder_input")
    hidden_decoder = tf.layers.dense(decoder_input, 192, activation=tf.nn.relu)
    decoder_8 = tf.reshape(hidden_decoder, [-1, 8, 8, 3])

    # Todo: upscale and output
    decoder_16 = upscale_block(decoder_input)
    decoder_32 = upscale_block(decoder_16)
    decoder_output = tf.identity(decoder_32, name="decoder_output")

    #peak signal to noise ratio
    mse = tf.reduce_mean(tf.squared_difference(output, x))
    psnr_1 = 20 * log_10(tf.constant(255**2, dtype=tf.float32))
    psnr_2 = 10 * log_10(mse)
    psnr = psnr_1 - psnr_2

    with tf.name_scope('optimizer') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(psnr)


    with tf.Session() as session:

        for epoch in range(FLAGS.max_epoch_num):
            print('Epoch: ' + str(epoch))

            psnr_vals = []
            for i in range(train_num_examples // batch_size):
                batch_xs = train_data[i*batch_size:(i+1)*batch_size, :]
                train_psnr = session.run([psnr], {x: batch_xs})
                psnr_vals.append(train_psnr)
            avg_train_psnr = sum(psnr_vals) / len(psnr_vals)
            print('Train PSNR: ' + str(avg_train_psnr))

            psnr_vals = []
            for i in range(test_num_examples // batch_size):
                batch_xs = test_data[i*batch_size:(i+1)*batch_size, :]
                test_psnr = session.run([psnr], {x: batch_xs})
                psnr_vals.append(test_psnr)
            avg_test_psnr = sum(psnr_vals) / len(psnr_vals)
            print('Test PSNR: ' + str(avg_test_psnr))
            print('--------------------')




if __name__ == "__main__":
    tf.app.run()