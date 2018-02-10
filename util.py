import numpy as np

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

def update_keep_prob(keep_prob, rate):
    new_keep_prob = keep_prob + rate
    if (new_keep_prob > 1):
        new_keep_prob = 1
    return new_keep_prob