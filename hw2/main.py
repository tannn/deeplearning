import tensorflow as tf
import os
import numpy as np
from model import *
from util import *

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/02/EMODB-German/', 'directory where EMODB-German is located')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num_german', 6, '')
flags.DEFINE_integer('max_epoch_num_english', 50, '')
FLAGS = flags.FLAGS

def main(argv):
    # load text file
    with open("~/path.txt", "r") as f: 
        path_string = f.read().split(sep='\n')[0]

    save_dir = '/work/' + path_string + '/homework02/logs'

    x = tf.placeholder(shape=[None, 16641], dtype=tf.float32, name='input_placeholder')
    x_reshaped = tf.reshape(x, [-1, 129, 129, 1])
    filter_sizes = [16, 32, 64]
    conv_x = my_conv_block(x_reshaped, filter_sizes)
    flat = tf.reshape(conv_x, [-1, 17*17*filter_sizes[2]])


    y = tf.placeholder(tf.float32, [None, 7], name='label')

    output_dense_german = dense_block(flat, language="german")
    output_german = tf.identity(output_dense_german, name='output')
    cross_entropy_german  = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_german)
    confusion_matrix_op_german = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output_german, axis=1), num_classes=7)
    sum_cross_entropy_german = tf.reduce_mean(cross_entropy_german)

    output_dense_english = dense_block(flat, language="english")
    output_english = tf.identity(output_dense_english, name='output')
    cross_entropy_english  = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_english)
    confusion_matrix_op_english = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output_english, axis=1), num_classes=7)
    sum_cross_entropy_english = tf.reduce_mean(cross_entropy_english)
    


    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op_english = optimizer.minimize(cross_entropy_english, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "dense_block"))
    train_op_german = optimizer.minimize(cross_entropy_german, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "dense_block"))
    saver = tf.train.Saver()
    
    optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "optimizer")
    dense_vars_german = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "dense_block_german")
    dense_vars_english = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "dense_block_english")
    session.run(tf.variables_initializer(optimizer_vars + dense_vars_german, name='init'))

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        # run training
        batch_size = FLAGS.batch_size

        best_epoch = -1
        # grace = 15
        # counter = 0

        train_images_list, train_labels_list, test_images_list, test_labels_list = load_data(FLAGS.data_dir)

        for fold in range(4):
            train_images = train_images_list[fold]
            train_labels = train_labels_list[fold]
            test_images = test_images_list[fold]
            test_labels = test_labels_list[fold]
            train_num_examples = train_images.shape[0]
            test_num_examples = test_images.shape[0]

            print('Fold ' + str(fold))

            for epoch in range(FLAGS.max_epoch_num_german):
                print('Epoch: ' + str(epoch))

                # run gradient steps and report mean loss on train data
                ce_vals = []
                for i in range(train_num_examples // batch_size):
                    batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = train_labels.reshape(-1,7)[i*batch_size:(i+1)*batch_size, :]    
                    _, train_ce = session.run([train_op_german, sum_cross_entropy_german], {x: batch_xs, y: batch_ys})
                    ce_vals.append(train_ce)
                avg_train_ce = sum(ce_vals) / len(ce_vals)
                print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))


                # report mean test loss
                ce_vals = []
                conf_mxs = []
                for i in range(test_num_examples // batch_size):
                    batch_xs = test_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = test_labels.reshape(-1,7)[i*batch_size:(i+1)*batch_size, :]    
                    test_ce, conf_matrix = session.run([sum_cross_entropy_german, confusion_matrix_op_german], {x: batch_xs, y: batch_ys})
                    ce_vals.append(test_ce)
                    conf_mxs.append(conf_matrix)
                avg_test_ce = sum(ce_vals) / len(ce_vals)
                print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
                print('TEST CONFUSION MATRIX:')
                conf_matrix = sum(conf_mxs)
                print(str(conf_matrix))
                correct = 0
                for i in range(7):
                    correct += conf_matrix[i, i]
                test_class_rate = float(correct) / sum(sum(conf_matrix))
                    
                print('TEST CLASSIFICATION RATE: ' + str(test_class_rate))
                print('--------------------')

                ##### TODO: Add a saver line here
                ##### can copy and paste the one from the old code
                ##### just use gitk and look like 10 or 20 commits back

                best_path_prefix = saver.save(session, os.path.join(save_dir, "homework_02"))
           
            print('--------------------')

        data_dir = '/work/cse496dl/shared/homework/02/SAVEE-British/'
        train_images_list, train_labels_list, test_images_list, test_labels_list = load_data(data_dir)
        session.run(tf.variables_initializer(optimizer_vars + dense_vars_english, name='init'))



if __name__ == "__main__":
    tf.app.run()