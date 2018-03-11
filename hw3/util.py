import tensorflow as tf 
import numpy as np

def log_10(tensor):
    return tf.log(tensor) / tf.log(tf.constant(10.))

def split_data(data, proportion=0.1):
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`
    
    Args:
        - data: numpy array of data, to be split along the first axis
        - proportion: a float less than 1  
    """
    size = data.shape[0]
    np.random.seed(69)
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[s[:split_idx]], data[s[split_idx:]]