import tensorflow as tf

def my_conv_block(inputs, filters):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - filters: iterable of ints of length 3
    """
    with tf.name_scope('conv_block') as scope:
        first_conv = tf.layers.Conv2D(filters[0], 3, 1, padding='same')
        second_conv = tf.layers.Conv2D(filters[1], 3, 1, padding='same')
        third_conv = tf.layers.Conv2D(filters[2], 3, 1, padding='same')
        pool = tf.layers.MaxPooling2D(2, 2, padding='same')
        output_tensor = pool(third_conv(pool(second_conv(pool(first_conv(inputs))))))
        layer_list = [first_conv, second_conv, third_conv, pool]
        block_parameter_num = sum(map(lambda layer: layer.count_params(), layer_list))
        print('Number of parameters in conv block: ' + str(block_parameter_num))

        return output_tensor

