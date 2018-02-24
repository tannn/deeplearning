import tensorflow as tf

def my_conv_block(inputs, filters):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - filters: iterable of ints of length 3
    """
    with tf.name_scope('conv_block') as scope:
        first_conv = tf.layers.conv2d(inputs, filters[0], 3, 1, padding='same')
        pool_1 = tf.layers.max_pooling2d(first_conv, 2, 2, padding='same')
        second_conv = tf.layers.conv2d(pool_1, filters[1], 3, 1, padding='same')
        pool_2 = tf.layers.max_pooling2d(second_conv, 2, 2, padding='same')
        third_conv = tf.layers.conv2d(pool_2, filters[2], 3, 1, padding='same')
        pool_3 = tf.layers.max_pooling2d(third_conv, 2, 2, padding='same')
        return pool_3

def dense_block(inputs, language):
    with tf.name_scope('dense_block_' + language) as scope:
        hidden_1 = tf.layers.dense(inputs, 512, activation=tf.nn.relu, name=language + "_hidden_1")
        hidden_2 = tf.layers.dense(hidden_1, 128, activation=tf.nn.relu, name=language + "_hidden_2")
        output_layer = tf.layers.dense(hidden_2, 7, name=language + "_output")
        return output_layer

def optimizer_block(language, layer, label, rate):
    with tf.name_scope('optimizer_'+ language) as scope:
        print(list(layer.get_shape()))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=layer)
        optimizer = tf.train.AdamOptimizer(learning_rate=rate)
        if (language=='english'):
            variables_to_train = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "dense_block_english")
            train_op = optimizer.minimize(cross_entropy, var_list=variables_to_train)
        else:
            train_op = optimizer.minimize(cross_entropy)
        
        return optimizer, cross_entropy, train_op