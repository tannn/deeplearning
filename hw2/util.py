import tensorflow as tf 

def load_sesh(pathname, meta_file, network_name):
    session = tf.Session()
    saver = tf.train.import_meta_graph(pathname + meta_file)
    saver.restore(session, pathname + network_name)
    print(session.graph.get_operations())