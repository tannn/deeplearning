import tensorflow as tf
import numpy as np
import os
from util import *
from sklearn.preprocessing import OneHotEncoder

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/01/', 'directory where MNIST is located')
flags.DEFINE_string('username', 'bgeren', '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 50, '')
FLAGS = flags.FLAGS

def main(argv):
    # load data

    save_dir = '/work/cse496dl/' + FLAGS.username + '/homework01/logs'

    train_images = np.load(FLAGS.data_dir + 'fmnist_train_data.npy')
    train_labels = np.load(FLAGS.data_dir + 'fmnist_train_labels.npy')

    valid_images, train_images, valid_labels, train_labels = split_data(train_images, train_labels, 0.1)

    # split into train and validate
    train_num_examples = train_images.shape[0]
    valid_num_examples = valid_images.shape[0]

    keep_prob = 0.8
    grace = 15
    counter = 0

    keep_prob_updater_rate = 0.05

    # specify the network
    x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    x_normalized = x / 255

    with tf.name_scope('linear_model') as scope:

        hidden_1 = tf.layers.dense(x_normalized,
                                 420,
                                 activation=tf.nn.relu,
                                 name='hidden_layer_1')

        dropout_1 = tf.nn.dropout(hidden_1, keep_prob)
        keep_prob = update_keep_prob(keep_prob, keep_prob_updater_rate)

        hidden_2 = tf.layers.dense(dropout_1,
                                 225,
                                 activation=tf.nn.relu,
                                 name='hidden_layer_2')
        
        dropout_2 = tf.nn.dropout(hidden_2, keep_prob)
        keep_prob = update_keep_prob(keep_prob, keep_prob_updater_rate)

        hidden_3 = tf.layers.dense(dropout_2,
                                 121,
                                 activation=tf.nn.relu,
                                 name='hidden_layer_3')

        dropout_3 = tf.nn.dropout(hidden_3, keep_prob)
        keep_prob = update_keep_prob(keep_prob, keep_prob_updater_rate)

        hidden_4 = tf.layers.dense(dropout_3,
                                 65,
                                 activation=tf.nn.relu,
                                 name='hidden_layer_4')
        
        dropout_4 = tf.nn.dropout(hidden_4, keep_prob)
        keep_prob = update_keep_prob(keep_prob, keep_prob_updater_rate)

        hidden_5 = tf.layers.dense(dropout_4,
                                 35,
                                 activation=tf.nn.relu,
                                 name='hidden_layer_5')

        dropout_5 = tf.nn.dropout(hidden_5, keep_prob)
        keep_prob = update_keep_prob(keep_prob, keep_prob_updater_rate)

        hidden_6 = tf.layers.dense(dropout_5,
                                 19,
                                 activation=tf.nn.relu,
                                 name='hidden_layer_6')
        
        dropout_6 = tf.nn.dropout(hidden_6, keep_prob)


    output = tf.layers.dense(dropout_6,
                             10,
                             name='output')
    
    tf.identity(output, name='output')

    #     hidden_1 = tf.layers.dense(x_normalized,
    #                              420,
    #                              activation=tf.nn.relu,
    #                              name='hidden_layer_1')

    #     hidden_2 = tf.layers.dense(hidden_1,
    #                              225,
    #                              activation=tf.nn.relu,
    #                              name='hidden_layer_2')

    #     hidden_3 = tf.layers.dense(hidden_2,
    #                              121,
    #                              activation=tf.nn.relu,
    #                              name='hidden_layer_3')

    #     hidden_4 = tf.layers.dense(hidden_3,
    #                              65,
    #                              activation=tf.nn.relu,
    #                              name='hidden_layer_4')

    #     hidden_5 = tf.layers.dense(hidden_4,
    #                              35,
    #                              activation=tf.nn.relu,
    #                              name='hidden_layer_5')

    #     hidden_6 = tf.layers.dense(hidden_5,
    #                              19,
    #                              activation=tf.nn.relu,
    #                              name='hidden_layer_6')
        
    # output = tf.layers.dense(hidden_6,
    #                          10,
    #                          name='output')


    # define classification loss
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)

    sum_cross_entropy = tf.reduce_mean(cross_entropy)

    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)

    # accuracy = tf.metrics.accuracy(labels=y, predictions=output)

    # set up training and saving functionality
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(cross_entropy)
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # run training
        batch_size = FLAGS.batch_size

        onehot_encoder = OneHotEncoder(sparse=False)

        best_class_rate = float("-inf")
        best_epoch = -1
        for epoch in range(FLAGS.max_epoch_num):
            print('Epoch: ' + str(epoch))

            # run gradient steps and report mean loss on train data
            ce_vals = []
            for i in range(train_num_examples // batch_size):
                batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
                batch_ys = onehot_encoder.fit_transform(train_labels.reshape(-1,1))[i*batch_size:(i+1)*batch_size, :]    
                _, train_ce = session.run([train_op, sum_cross_entropy], {x: batch_xs, y: batch_ys})
                ce_vals.append(train_ce)
            avg_train_ce = sum(ce_vals) / len(ce_vals)
            print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))


            # report mean validation loss
            ce_vals = []
            conf_mxs = []
            for i in range(valid_num_examples // batch_size):
                batch_xs = valid_images[i*batch_size:(i+1)*batch_size, :]
                batch_ys = onehot_encoder.fit_transform(valid_labels.reshape(-1,1))[i*batch_size:(i+1)*batch_size, :]    
                test_ce, conf_matrix = session.run([sum_cross_entropy, confusion_matrix_op], {x: batch_xs, y: batch_ys})
                ce_vals.append(test_ce)
                conf_mxs.append(conf_matrix)
            avg_valid_ce = sum(ce_vals) / len(ce_vals)
            print('VALIDATION CROSS ENTROPY: ' + str(avg_valid_ce))
            print('VALIDATION CONFUSION MATRIX:')
            conf_matrix = sum(conf_mxs)
            print(str(conf_matrix))
            correct = 0
            for i in range(10):
                correct += conf_matrix[i, i]
            class_rate = float(correct) / sum(sum(conf_matrix))
                
            print('VALIDATION CLASSIFICATION RATE: ' + str(class_rate))

            if (class_rate > best_class_rate):
                print('New best found!')
                best_train_loss = avg_train_ce
                best_valid_loss = avg_valid_ce
                best_epoch = epoch                
                best_path_prefix = saver.save(session, os.path.join(save_dir, "homework_1-0"))
                best_conf_mx = conf_matrix
                best_class_rate = class_rate
                counter = 0
            else:
                counter = counter + 1

            if counter >= grace:
                break

    print('BEST EPOCH: ' + str(best_epoch))
    print('TRAIN LOSS: ' + str(best_train_loss))
    print('VALIDATION LOSS: '  + str(best_valid_loss))
    print('CONFUSION MATRIX')
    print(str(best_conf_mx))
    print('BEST CLASSIFICATION RATE: ' + str(best_class_rate))

if __name__ == "__main__":
    tf.app.run()

    