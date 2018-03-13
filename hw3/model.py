import tensorflow as tf
import numpy as np

# Upscale and Downscale blocks taken from Paul Quint
def upscale_block(x, scale=2):
    """ conv2d_transpose """
    return tf.layers.conv2d_transpose(x, 3, 3, strides=(scale, scale), padding='same', activation=tf.nn.relu)

def downscale_block(x, scale=2):
    n, h, w, c = x.get_shape().as_list()
    return tf.layers.conv2d(x, np.floor(c * 1.25), 3, strides=scale, padding='same', activation=tf.nn.relu)

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
    flat = flatten(encoder_16)
    code = tf.layers.dense(flat, 100, activation=tf.nn.relu)

    decoder_input = tf.identity(code, name="decoder_input")
    hidden_decoder = tf.layers.dense(decoder_input, 768, activation=tf.nn.relu)
    decoder_16 = tf.reshape(hidden_decoder, [-1, 16, 16, 3])
    decoder_32 = upscale_block(decoder_16)
    decoder_output = tf.identity(decoder_32, name="decoder_output")

    return code, decoder_input, decoder_output
