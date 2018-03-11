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

    grace = 15
    counter = 0

    save_dir = '/work/' + path_string + '/homework03/logs'

    train_data = np.load(FLAGS.data_dir + 'cifar10_train_data.npy')
    test_data = np.load(FLAGS.data_dir + 'cifar10_test_data.npy')

    valid_images, train_images = split_data(train_data)

    train_num_examples = train_images.shape[0]
    test_num_examples = test_data.shape[0]
    valid_num_examples = valid_images.shape[0]

    print(train_data.shape)
    best_class_rate = float("-inf")

    #TODO: Reshape incoming data to be square

    x = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='encoder_input')

    #TODO: create the code as a flatten -> dense 

    decoder_output = autoencoder_network(x)

    #peak signal to noise ratio
    mse = tf.reduce_mean(tf.squared_difference(decoder_output, x))
    psnr_1 = 20 * log_10(tf.constant(255**2, dtype=tf.float32))
    psnr_2 = 10 * log_10(mse)
    negative_psnr = psnr_2 - psnr_1

    # Define saver tensor 
    saver = tf.train.Saver()

    with tf.name_scope('optimizer') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(negative_psnr)


    with tf.Session() as session:

        session.run(tf.global_variables_initializer())


        for epoch in range(FLAGS.max_epoch_num):
            print('Epoch: ' + str(epoch))

            psnr_vals = []
            for i in range(train_num_examples // batch_size):
                batch_xs = train_data[i*batch_size:(i+1)*batch_size, :]
                _, train_psnr = session.run([train_op, negative_psnr], {x: batch_xs})
                train_psnr *= -1
                psnr_vals.append(train_psnr)
            avg_train_psnr = sum(psnr_vals) / len(psnr_vals)
            print('Train PSNR: ' + str(avg_train_psnr))

            psnr_vals = []
            for i in range(test_num_examples // batch_size):
                batch_xs = test_data[i*batch_size:(i+1)*batch_size, :]
                test_psnr = session.run(negative_psnr, {x: batch_xs})
                test_psnr *= -1
                psnr_vals.append(test_psnr)
            avg_test_psnr = sum(psnr_vals) / len(psnr_vals)
            print('Test PSNR: ' + str(avg_test_psnr))


            # report mean validation loss
            psnr_vals = []
            for i in range(valid_num_examples // batch_size):
                batch_xs = valid_images[i*batch_size:(i+1)*batch_size, :]
                valid_psnr = session.run(negative_psnr, {x: batch_xs})
                valid_psnr *= -1
                psnr_vals.append(valid_psnr)
            avg_valid_psnr = sum(psnr_vals) / len(psnr_vals)
            print('Valid PSNR: ' + str(avg_valid_psnr))

            if (avg_valid_psnr > best_class_rate):
                print('New best found!')
                best_train_loss = avg_train_psnr
                best_valid_loss = avg_valid_psnr
                best_epoch = epoch
                counter = 0
            else:
                counter = counter + 1

            if counter >= grace:
                break

            print('--------------------')
            saver.save(session, os.path.join(save_dir, "maxcompression_encoder_homework_3-0"))
            saver.save(session, os.path.join(save_dir, "maxquality_encoder_homework_3-0"))
            saver.save(session, os.path.join(save_dir, "maxcompression_decoder_homework_3-0"))
            saver.save(session, os.path.join(save_dir, "maxquality_decoder_homework_3-0"))




if __name__ == "__main__":
    tf.app.run()