import tensorflow as tf 

def hidden_layers(x):
    with tf.name_scope('linear_model') as scope:

	    hidden_1 = tf.layers.dense(x,
	                             420,
	                             activation=tf.nn.relu,
	                             name='hidden_layer_1')

	    dropout_1 = tf.nn.dropout(hidden_1, keep_prob)
	    keep_prob = update_keep_prob(keep_prob, keep_prob_updater_rate)

	    hidden_2 = tf.layers.dense(dropout_1,
	                             225,
	                             activation=tf.nn.relu,
	                             name='hidden_layer_2')
	    
	    dropout_2 = tf.nn.dropout(hidden_2, keep_prob)
	    keep_prob = update_keep_prob(keep_prob, keep_prob_updater_rate)

	    hidden_3 = tf.layers.dense(dropout_2,
	                             121,
	                             activation=tf.nn.relu,
	                             name='hidden_layer_3')

	    dropout_3 = tf.nn.dropout(hidden_3, keep_prob)
	    keep_prob = update_keep_prob(keep_prob, keep_prob_updater_rate)

	    hidden_4 = tf.layers.dense(dropout_3,
	                             65,
	                             activation=tf.nn.relu,
	                             name='hidden_layer_4')
	    
	    dropout_4 = tf.nn.dropout(hidden_4, keep_prob)
	    keep_prob = update_keep_prob(keep_prob, keep_prob_updater_rate)

	    hidden_5 = tf.layers.dense(dropout_4,
	                             35,
	                             activation=tf.nn.relu,
	                             name='hidden_layer_5')

	    dropout_5 = tf.nn.dropout(hidden_5, keep_prob)
	    keep_prob = update_keep_prob(keep_prob, keep_prob_updater_rate)

	    hidden_6 = tf.layers.dense(dropout_5,
	                             19,
	                             activation=tf.nn.relu,
	                             name='hidden_layer_6')
	    
	    dropout_6 = tf.nn.dropout(hidden_6, keep_prob)

	    output_layer = tf.layers.dense(dropout_6,
	                         10,
	                         name='output_layer')

#     hidden_1 = tf.layers.dense(x_normalized,
#                              420,
#                              activation=tf.nn.relu,
#                              name='hidden_layer_1')

#     hidden_2 = tf.layers.dense(hidden_1,
#                              225,
#                              activation=tf.nn.relu,
#                              name='hidden_layer_2')

#     hidden_3 = tf.layers.dense(hidden_2,
#                              121,
#                              activation=tf.nn.relu,
#                              name='hidden_layer_3')

#     hidden_4 = tf.layers.dense(hidden_3,
#                              65,
#                              activation=tf.nn.relu,
#                              name='hidden_layer_4')

#     hidden_5 = tf.layers.dense(hidden_4,
#                              35,
#                              activation=tf.nn.relu,
#                              name='hidden_layer_5')

#     hidden_6 = tf.layers.dense(hidden_5,
#                              19,
#                              activation=tf.nn.relu,
#                              name='hidden_layer_6')
    
#	  output_layer = tf.layers.dense(hidden_6,
#                          10,
#                          name='output_layer')

	    return output_layer

