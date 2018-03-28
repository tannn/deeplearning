import tensorflow as tf
import numpy as np

# Upscale and Downscale blocks taken from Paul Quint
def upscale_block_with_l2(x, scale=2):
    """ conv2d_transpose """
    return tf.layers.conv2d_transpose(x, 3, 3, strides=(scale, scale), 
                                               padding='same',
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                               bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.), 
                                               activation=tf.nn.relu)
def upscale_block(x, scale=2):
    """ conv2d_transpose """
    return tf.layers.conv2d_transpose(x, 3, 3, strides=(scale, scale), padding='same', activation=tf.nn.relu)

def downscale_block(x, scale=2):
    n, h, w, c = x.get_shape().as_list()
    return tf.layers.conv2d(x, np.floor(c * 1.25), 3, strides=scale, padding='same', activation=tf.nn.relu)

def downscale_block_with_l2(x, scale=2):
    n, h, w, c = x.get_shape().as_list()
    return tf.layers.conv2d(x, np.floor(c * 1.25), 3, strides=scale, 
                                                      padding='same',
                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                                      bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.), 
                                                      activation=tf.nn.relu)

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

def autoencoder_network(x):
    encoder_16 = downscale_block(x)
    encoder_8 = downscale_block(encoder_16)
    flat = flatten(encoder_8)
    code = tf.layers.dense(flat, 100, activation=tf.nn.relu)

    decoder_input = tf.identity(code, name="decoder_input")
    hidden_decoder = tf.layers.dense(decoder_input, 192, activation=tf.nn.relu)
    decoder_8 = tf.reshape(hidden_decoder, [-1, 8, 8, 3])
    decoder_16 = upscale_block(decoder_8)
    decoder_32 = upscale_block(decoder_16)
    decoder_output = tf.identity(decoder_32, name="decoder_output")

    return code, decoder_input, decoder_output


def autoencoder_network_with_l2(x):
    encoder_16 = downscale_block_with_l2(x)
    encoder_8 = downscale_block_with_l2(encoder_16)
    flat = flatten(encoder_8)
    code = tf.layers.dense(flat, 100, activation=tf.nn.relu)

    decoder_input = tf.identity(code, name="decoder_input")
    hidden_decoder = tf.layers.dense(decoder_input, 192, activation=tf.nn.relu,
                                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                                         bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.))
    decoder_8 = tf.reshape(hidden_decoder, [-1, 8, 8, 3])
    decoder_16 = upscale_block_with_l2(decoder_8)
    decoder_32 = upscale_block_with_l2(decoder_16)
    decoder_output = tf.identity(decoder_32, name="decoder_output")

    return code, decoder_input, decoder_output


def fake_autoencoder(x):
    flat = flatten(x)
    hidden_1 = tf.layers.dense(flat, 3072, activation=tf.nn.relu, kernel_initializer=tf.ones_initializer)
    hidden_1 = hidden_1 / 3072
    code = tf.cast(hidden_1, dtype=tf.uint8)
    decoder_input = tf.identity(code, name="decoder_input")
    decoder = tf.cast(decoder_input, dtype=tf.float32)
    hidden_2 = tf.layers.dense(decoder, 3072, activation=tf.nn.relu, kernel_initializer=tf.ones_initializer)
    hidden_2 = hidden_2 / 3072
    square = tf.reshape(hidden_2, [-1, 32, 32, 3])
    decoder_output = tf.identity(square, name="decoder_output")

    return code, decoder_input, decoder_output

def autoencoder_network_max_compression(x):
    flat = flatten(x)
    code = tf.cast(flat, dtype=tf.uint8)
    code = code // 2
    code = code * 2
    decoder_input = tf.identity(code, name="decoder_input")
    decoder = tf.cast(decoder_input, dtype=tf.float32)
    square = tf.reshape(decoder, [-1, 32, 32, 3])
    decoder_output = tf.identity(square, name="decoder_output")

    return code, decoder_input, decoder_output

