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

    #TODO: create the code as a flatten -> dense 

    decoder_output = autoencoder_network(x)

    #peak signal to noise ratio
    mse = tf.reduce_mean(tf.squared_difference(decoder_output, x))
    psnr_1 = 20 * log_10(tf.constant(255**2, dtype=tf.float32))
    psnr_2 = 10 * log_10(mse)
    psnr = psnr_1 - psnr_2

    # Define saver tensor 
    saver = tf.train.Saver()

    with tf.name_scope('optimizer') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(psnr)


    with tf.Session() as session:

        session.run(tf.global_variables_initializer())


        for epoch in range(FLAGS.max_epoch_num):
            print('Epoch: ' + str(epoch))

            psnr_vals = []
            for i in range(train_num_examples // batch_size):
                batch_xs = train_data[i*batch_size:(i+1)*batch_size, :]
                train_psnr = session.run(psnr, {x: batch_xs})
                psnr_vals.append(train_psnr)
            avg_train_psnr = sum(psnr_vals) / len(psnr_vals)
            print('Train PSNR: ' + str(avg_train_psnr))

            psnr_vals = []
            for i in range(test_num_examples // batch_size):
                batch_xs = test_data[i*batch_size:(i+1)*batch_size, :]
                test_psnr = session.run(psnr, {x: batch_xs})
                psnr_vals.append(test_psnr)
            avg_test_psnr = sum(psnr_vals) / len(psnr_vals)
            print('Test PSNR: ' + str(avg_test_psnr))
            print('--------------------')

            path = saver.save(session, os.path.join(save_dir, "cifar_homework_1-0"))




if __name__ == "__main__":
    tf.app.run()