import tensorflow as tf
import numpy as np
import os

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/01/', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '/work/cse496dl/bgeren/hackathon3/logs', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 100, '')
FLAGS = flags.FLAGS


def main(argv):
    # load data
    train_images = np.load(FLAGS.data_dir + 'fmnist_train_data.npy')
    train_labels = np.load(FLAGS.data_dir + 'fmnist_train_labels.npy')

    valid_images, train_images, valid_labels, train_labels = split_data(train_images, train_labels, 0.1)

    # split into train and validate
    ## TODO
    train_num_examples = train_images.shape[0]
    valid_num_examples = valid_images.shape[0]
    test_num_examples = test_images.shape[0]

    # specify the network
    x = tf.placeholder(tf.float32, [None, 784], name='data')
    with tf.name_scope('linear_model') as scope:
        hidden = tf.layers.dense(x,
                                 400,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                 activation=tf.nn.relu,
                                 name='hidden_layer')
        output = tf.layers.dense(hidden,
                                 10,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                 name='output_layer')
    # define classification loss
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # this is the weight of the regularization part of the final loss
    REG_COEFF = 0.001
    # this value is what we'll pass to `minimize`
    total_loss = cross_entropy + REG_COEFF * sum(regularization_losses)
    sum_cross_entropy = tf.reduce_mean(cross_entropy)

    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)


    # set up training and saving functionality
    # global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(total_loss)
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
            for i in range(valid_num_examples // batch_size):
                batch_xs = valid_images[i*batch_size:(i+1)*batch_size, :]
                batch_ys = valid_labels[i*batch_size:(i+1)*batch_size, :]       
                valid_ce, _ = session.run([sum_cross_entropy, y], {x: batch_xs, y: batch_ys})
                ce_vals.append(valid_ce)
            avg_valid_ce = sum(ce_vals) / len(ce_vals)
            print('VALIDATION CROSS ENTROPY: ' + str(avg_valid_ce))
            

            # report mean test loss
            ce_vals = []
            conf_mxs = []
            for i in range(test_num_examples // batch_size):
                batch_xs = test_images[i*batch_size:(i+1)*batch_size, :]
                batch_ys = test_labels[i*batch_size:(i+1)*batch_size, :]
                test_ce, conf_matrix = session.run([sum_cross_entropy, confusion_matrix_op], {x: batch_xs, y: batch_ys})
                ce_vals.append(test_ce)
                conf_mxs.append(conf_matrix)
            avg_test_ce = sum(ce_vals) / len(ce_vals)
            print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
            print('TEST CONFUSION MATRIX:')


            print(str(sum(conf_mxs)))

            if (avg_valid_ce < best_valid_loss):
                best_train_loss = avg_train_ce
                best_valid_loss = avg_valid_ce
                best_test_loss = avg_test_ce
                best_epoch = epoch                
                # best_path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, "mnist_inference"), global_step=global_step_tensor)

    # # Clear the graph
    # tf.reset_default_graph()
    # session = tf.Session()
    # graph = session.graph

    # # loading the meta graph re-creates the graph structure in the current session, and restore initializes saved variables
    # saver = tf.train.import_meta_graph(best_path_prefix + '.meta')
    # saver.restore(session, best_path_prefix)
    # x = graph.get_tensor_by_name('data_placeholder:0')
    # output = graph.get_tensor_by_name('linear_model/model_output:0')

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
    np.random.seed(42)
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[s[:split_idx]], data[s[split_idx:]], labels[s[:split_idx]], labels[s[split_idx:]]

if __name__ == "__main__":
    tf.app.run()

