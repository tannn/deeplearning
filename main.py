import tensorflow as tf
import numpy as np
import os

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/01/', 'directory where MNIST is located')
flags.DEFINE_string('username', 'bgeren', '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 100, '')
FLAGS = flags.FLAGS

def main(argv):
    # load data

    save_dir = '/work/cse496dl/' + FLAGS.username + '/homework01/logs'

    train_images = np.load(FLAGS.data_dir + 'fmnist_train_data.npy')
    train_labels = np.load(FLAGS.data_dir + 'fmnist_train_labels.npy')

    train_images /= 255

    valid_images, train_images, valid_labels, train_labels = split_data(train_images, train_labels, 0.1)

    # split into train and validate
    train_num_examples = train_images.shape[0]
    valid_num_examples = valid_images.shape[0]

    keep_prob = 0.8

    # specify the network
    x = tf.placeholder(tf.float32, [None, 784], name='data')
    with tf.name_scope('linear_model') as scope:

        dropout_1 = tf.nn.dropout(x, keep_prob)

        hidden_1 = tf.layers.dense(dropout_1,
                                 256,
                                 activation=tf.nn.relu,
                                 name='hidden_layer_1')

        dropout_2 = tf.nn.dropout(hidden_1, keep_prob)

        hidden_2 = tf.layers.dense(dropout_2,
                                 256,
                                 activation=tf.nn.relu,
                                 name='hidden_layer_2')
        
        dropout_3 = tf.nn.dropout(hidden_2, keep_prob)

        output = tf.layers.dense(dropout_3,
                                 10,
                                 name='output_layer')
    # define classification loss
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)

    sum_cross_entropy = tf.reduce_mean(cross_entropy)

    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)


    # set up training and saving functionality
    # global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(cross_entropy)
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # run training
        batch_size = FLAGS.batch_size

        best_valid_loss = float("inf")
        best_epoch = -1
        for epoch in range(FLAGS.max_epoch_num):
            print('Epoch: ' + str(epoch))

            # run gradient steps and report mean loss on train data
            ce_vals = []
            for i in range(train_num_examples // batch_size):
                batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
                batch_ys = train_labels[i*batch_size:(i+1)*batch_size, :]       
                _, train_ce = session.run([train_op, sum_cross_entropy], {x: batch_xs, y: batch_ys})
                ce_vals.append(train_ce)
            avg_train_ce = sum(ce_vals) / len(ce_vals)
            print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))


            # report mean validation loss
            ce_vals = []
            conf_mxs = []
            for i in range(valid_num_examples // batch_size):
                batch_xs = valid_images[i*batch_size:(i+1)*batch_size, :]
                batch_ys = valid_labels[i*batch_size:(i+1)*batch_size, :]       
                test_ce, conf_matrix = session.run([sum_cross_entropy, confusion_matrix_op], {x: batch_xs, y: batch_ys})
                ce_vals.append(test_ce)
                conf_mxs.append(conf_matrix)
            avg_valid_ce = sum(ce_vals) / len(ce_vals)
            print('VALIDATION CROSS ENTROPY: ' + str(avg_valid_ce))
            print('TEST CONFUSION MATRIX:')
            print(str(sum(conf_mxs)))


            if (avg_valid_ce < best_valid_loss):
                best_train_loss = avg_train_ce
                best_valid_loss = avg_valid_ce
                best_epoch = epoch                
                best_path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, "homework_01"), global_step=global_step_tensor)

    print('BEST EPOCH: ' + str(best_epoch))
    print('TRAIN LOSS: ' + str(best_train_loss))
    print('VALIDATION LOSS: '  + str(best_valid_loss))
    print('TEST LOSS: ' + str(best_test_loss))


def split_data(data, labels, proportion):
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`
    
    Args:
        - data: numpy array of data, to be split along the first axis
        - labels: numpy array of the labels
        - proportion: a float less than 1  
    """
    size = data.shape[0]
    np.random.seed(69)
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[s[:split_idx]], data[s[split_idx:]], labels[s[:split_idx]], labels[s[split_idx:]]

if __name__ == "__main__":
    tf.app.run()


