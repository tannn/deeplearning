import tensorflow as tf
import os
import numpy as np
from model import *
from util import *

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/02/EMODB-German/', 'directory where EMODB-German is located')
flags.DEFINE_string('username', 'bgeren', '')
flags.DEFINE_string('workdir', 'cse496dl', '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 100, '')
FLAGS = flags.FLAGS

def main(argv):
    # load text file
    with open("path.txt", "r") as f: 
        path_string = f.read().split(sep='\n')[0]

    save_dir = '/work/' + path_string + '/homework02/logs'

    x = tf.placeholder(shape=[None, 16641], dtype=tf.float32, name='input_placeholder')
    x_reshaped = tf.reshape(x, [-1, 129, 129, 1])
    filter_sizes = [16, 32, 64]
    conv_x = my_conv_block(x_reshaped, filter_sizes)
    flat = tf.reshape(conv_x, [-1, 17*17*filter_sizes[2]])
    output = dense_block(flat)

    y = tf.placeholder(tf.float32, [None, 7], name='label')
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)

    sum_cross_entropy = tf.reduce_mean(cross_entropy)

    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=7)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(cross_entropy)
    saver = tf.train.Saver()
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # run training
        batch_size = FLAGS.batch_size

        best_epoch = -1
        grace = 15
        counter = 0

        train_images_list, train_labels_list, test_images_list, test_labels_list, valid_images_list, valid_labels_list = load_data(FLAGS.data_dir)

        for fold in range(4):
            train_images = train_images_list[fold]
            train_labels = train_labels_list[fold]
            valid_images = valid_images_list[fold]
            valid_labels = valid_labels_list[fold]
            test_images = test_images_list[fold]
            test_labels = test_labels_list[fold]
            train_num_examples = train_images.shape[0]
            valid_num_examples = valid_images.shape[0]
            test_num_examples = test_images.shape[0]
            best_valid_class_rate = float("-inf")

            print('Fold' + str(fold))

            for epoch in range(FLAGS.max_epoch_num):
                print('Epoch: ' + str(epoch))

                # run gradient steps and report mean loss on train data
                ce_vals = []
                for i in range(train_num_examples // batch_size):
                    batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = train_labels.reshape(-1,7)[i*batch_size:(i+1)*batch_size, :]    
                    _, train_ce = session.run([train_op, sum_cross_entropy], {x: batch_xs, y: batch_ys})
                    ce_vals.append(train_ce)
                avg_train_ce = sum(ce_vals) / len(ce_vals)
                print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))


                # report mean validation loss
                ce_vals = []
                conf_mxs = []
                for i in range(valid_num_examples // batch_size):
                    batch_xs = valid_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = valid_labels.reshape(-1,7)[i*batch_size:(i+1)*batch_size, :]    
                    valid_ce, conf_matrix = session.run([sum_cross_entropy, confusion_matrix_op], {x: batch_xs, y: batch_ys})
                    ce_vals.append(valid_ce)
                    conf_mxs.append(conf_matrix)
                avg_valid_ce = sum(ce_vals) / len(ce_vals)
                print('VALIDATION CROSS ENTROPY: ' + str(avg_valid_ce))
                print('VALIDATION CONFUSION MATRIX:')
                conf_matrix = sum(conf_mxs)
                print(str(conf_matrix))
                correct = 0
                for i in range(7):
                    correct += conf_matrix[i, i]
                valid_class_rate = float(correct) / sum(sum(conf_matrix))
                    
                print('VALIDATION CLASSIFICATION RATE: ' + str(valid_class_rate))

                # report mean test loss
                ce_vals = []
                conf_mxs = []
                for i in range(test_num_examples // batch_size):
                    batch_xs = test_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = test_labels.reshape(-1,7)[i*batch_size:(i+1)*batch_size, :]    
                    test_ce, conf_matrix = session.run([sum_cross_entropy, confusion_matrix_op], {x: batch_xs, y: batch_ys})
                    ce_vals.append(test_ce)
                    conf_mxs.append(conf_matrix)
                avg_valid_ce = sum(ce_vals) / len(ce_vals)
                print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
                print('TEST CONFUSION MATRIX:')
                conf_matrix = sum(conf_mxs)
                print(str(conf_matrix))
                correct = 0
                for i in range(7):
                    correct += conf_matrix[i, i]
                test_class_rate = float(correct) / sum(sum(conf_matrix))
                    
                print('TEST CLASSIFICATION RATE: ' + str(test_class_rate))


                if (valid_class_rate > best_valid_class_rate):
                    print('New best found!')
                    best_train_loss = avg_train_ce
                    best_valid_loss = avg_valid_ce
                    best_epoch = epoch                
                    best_path_prefix = saver.save(session, os.path.join(save_dir, "homework_1-0.fold" + str(fold)))
                    best_conf_mx = conf_matrix
                    best_valid_class_rate = valid_class_rate
                    best_test_class_rate = test_class_rate
                    counter = 0
                else:
                    counter = counter + 1

                if counter >= grace:
                    break
                print('--------------------')

            ###### TODO: add a print of all the best results
            ###### mirror it after what we did last homeowrk
            print('BEST EPOCH: ' + str(best_epoch))
            print('BEST TRAIN LOSS: ' + str(best_train_loss))
            print('BEST VALIDATION LOSS: ' + str(best_valid_loss))
            print('CONFUSION MATRIX')
            print(str(best_conf_mx))
            print('BEST TEST CLASSIFICATION RATE: ' + str(best_test_class_rate))
            print('BEST VALIDATION CLASSIFICATION RATE: ' + str(best_valid_class_rate))
            print('--------------------')
            print('--------------------')





if __name__ == "__main__":
    tf.app.run()