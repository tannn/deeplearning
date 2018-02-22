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

    train_images_list = [train_images_1, train_images_2, train_images_3, train_images_4]
    train_labels_list = [train_labels_1, train_labels_2, train_labels_3, train_labels_4]
    test_images_list = [test_images_1, test_images_2, test_images_3, test_images_4]
    test_labels_list = [test_labels_1, test_labels_2, test_labels_3, test_labels_4]
    valid_images_list = [valid_images_1, valid_images_2, valid_images_3, valid_images_4]
    valid_labels_list = [valid_labels_1, valid_labels_2, valid_labels_3, valid_labels_4]

    x = tf.placeholder(shape=[None, 16641], dtype=tf.float32, name='input_placeholder')
    x_reshaped = tf.reshape(x, [-1, 129, 129, 1])
    filter_sizes = [16, 32, 64]
    conv_x = my_conv_block(x_reshaped, filter_sizes)
    flat = tf.reshape(x_reshaped, [-1, 129*129*filter_sizes[2]])
    output = dense_block(flat)

    y = tf.placeholder(tf.float32, [None, 10], name='label')
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)

    sum_cross_entropy = tf.reduce_mean(cross_entropy)

    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(cross_entropy)
    saver = tf.train.Saver()
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # run training
        batch_size = FLAGS.batch_size

        best_class_rate = float("-inf")
        best_epoch = -1

        for fold in range(4):
            ###### TODO: I feel like we don't need to split the data because it is already split for us
            ####### we can use the data called "test" as our validation set, so we don't need to make a new validation set
            ###### could be wrong though
            train_images = train_images_list[fold]
            train_labels = train_labels_list[fold]
            valid_images = valid_images_list[fold]
            valid_labels = valid_labels_list[fold]
            train_num_examples = train_images.shape[0]
            valid_num_examples = valid_images.shape[0]


            for epoch in range(FLAGS.max_epoch_num):
                print('Epoch: ' + str(epoch))

                # run gradient steps and report mean loss on train data
                ce_vals = []
                for i in range(train_num_examples // batch_size):
                    batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = train_labels.reshape(-1,1)[i*batch_size:(i+1)*batch_size, :]    
                    _, train_ce = session.run([train_op, sum_cross_entropy], {x: batch_xs, y: batch_ys})
                    ce_vals.append(train_ce)
                avg_train_ce = sum(ce_vals) / len(ce_vals)
                print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))


                # report mean validation loss
                ce_vals = []
                conf_mxs = []
                for i in range(valid_num_examples // batch_size):
                    batch_xs = valid_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = valid_labels.reshape(-1,1)[i*batch_size:(i+1)*batch_size, :]    
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






if __name__ == "__main__":
    tf.app.run()