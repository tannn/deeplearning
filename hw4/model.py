# setup RNN

def rnn_block():
	outputs, state = tf.nn.dynamic_rnn(lstm_cell, word_embeddings,
	                                   initial_state=initial_state,
	                                   dtype=tf.float32)
	print("The outputs over all timesteps: "+ str(outputs))
	print("The final state of the LSTM layer: " + str(state))