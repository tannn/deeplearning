import tensorflow as tf 
import numpy as np

def load_sesh(pathname, meta_file, network_name):
    session = tf.Session()
    saver = tf.train.import_meta_graph(pathname + meta_file)
    saver.restore(session, pathname + network_name)
    print(session.graph.get_operations())

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

def load_data(data_dir):
    train_images_1 = np.load(data_dir + 'train_x_1.npy')
    train_labels_1 = np.load(data_dir + 'train_y_1.npy')

    test_images_1 = np.load(data_dir + 'test_x_1.npy')
    test_labels_1 = np.load(data_dir + 'test_y_1.npy')

    train_images_2 = np.load(data_dir + 'train_x_2.npy')
    train_labels_2 = np.load(data_dir + 'train_y_2.npy')

    test_images_2 = np.load(data_dir + 'test_x_2.npy')
    test_labels_2 = np.load(data_dir + 'test_y_2.npy')

    train_images_3 = np.load(data_dir + 'train_x_3.npy')
    train_labels_3 = np.load(data_dir + 'train_y_3.npy')

    test_images_3 = np.load(data_dir + 'test_x_3.npy')
    test_labels_3 = np.load(data_dir + 'test_y_3.npy')

    train_images_4 = np.load(data_dir + 'train_x_4.npy')
    train_labels_4 = np.load(data_dir + 'train_y_4.npy')

    test_images_4 = np.load(data_dir + 'test_x_4.npy')
    test_labels_4 = np.load(data_dir + 'test_y_4.npy')

    train_images_list = [train_images_1, train_images_2, train_images_3, train_images_4]
    train_labels_list = [train_labels_1, train_labels_2, train_labels_3, train_labels_4]
    test_images_list = [test_images_1, test_images_2, test_images_3, test_images_4]
    test_labels_list = [test_labels_1, test_labels_2, test_labels_3, test_labels_4]

    return train_images_list, train_labels_list, test_images_list, test_labels_list