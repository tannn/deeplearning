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

def dense_block(inputs):
    hidden_1 = tf.layers.dense(inputs, 512, activation=tf.nn.relu)
    hidden_2 = tf.layers.dense(hidden_1, 128, activation=tf.nn.relu)
    output_layer = tf.layers.dense(hidden_2, 7, name='output')
    return output_layer

