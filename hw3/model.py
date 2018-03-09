import tensorflow as tf
import numpy as np

def upscale_block(x, scale=2):
    """ conv2d_transpose """
    return tf.layers.conv2d_transpose(x, 3, 3, strides=(scale, scale), padding='same', activation=tf.nn.relu)

def downscale_block(x, scale=2):
    n, h, w, c = x.get_shape().as_list()
    return tf.layers.conv2d(x, np.floor(c * 1.25), 3, strides=scale, padding='same')

def my_conv_block(inputs, filters):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - filters: iterable of ints of length 3
    """
    with tf.name_scope('conv_block') as scope:
        first_conv = tf.layers.conv2d(inputs, filters[0], 3, 1, padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                      bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.), 
                                    )
        pool_1 = tf.layers.max_pooling2d(first_conv, 2, 2, padding='same')
        second_conv = tf.layers.conv2d(first_conv, filters[1], 3, 1, padding='same',
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                       bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.), 
                                    )
        pool_2 = tf.layers.max_pooling2d(second_conv, 2, 2, padding='same')
        third_conv = tf.layers.conv2d(pool_1, filters[2], 3, 1, padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                      bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.), 
                                    )
        pool_3 = tf.layers.max_pooling2d(third_conv, 2, 2, padding='same')
        return pool_3

def dense_block(inputs, language):
    with tf.name_scope('dense_block_' + language) as scope:
        hidden_1 = tf.layers.dense(inputs, 
                                   512, 
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                   activation=tf.nn.relu,
                                   name=language + "_hidden_1")
        hidden_2 = tf.layers.dense(hidden_1, 
                                   128, 
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                   activation=tf.nn.relu, 
                                   name=language + "_hidden_2")
        output_layer = tf.layers.dense(hidden_2, 7, name=language + "_output")
        return output_layer

def get_dim(inputs):
    """
    Flattens a tensor along all non-batch dimensions.
    This is correctly a NOP if the input is already flat.
    """
    if len(inputs.get_shape()) == 2:
        return inputs
    else:
        size = inputs.get_shape().as_list()[1:]
        return [-1, np.prod(np.array(size))]

def flatten(inputs):
    inputs_shape = get_dim(inputs)
    flat = tf.reshape(inputs, inputs_shape)
    return flat
