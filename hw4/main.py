import tensorflow as tf
import numpy as np
import sys
sys.path.append("/work/cse496dl/shared/hackathon/08")
import ptb_reader
from model import *
TIME_STEPS = 20
batch_size = 20
max_epoch = 50
DATA_DIR = '/work/cse496dl/shared/hackathon/08/ptbdata'

class PTBInput(object):
  """The input data.
  
  Code sourced from https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
  """

  def __init__(self, data, batch_size, num_steps, name=None):
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = ptb_reader.ptb_producer(
        data, batch_size, num_steps, name=name)

raw_data = ptb_reader.ptb_raw_data(DATA_DIR)
train_data, valid_data, test_data, _ = raw_data

train_input = PTBInput(train_data, batch_size, TIME_STEPS, name="TrainInput")
valid_input = PTBInput(valid_data, batch_size, TIME_STEPS, name="ValidInput")
test_input = PTBInput(test_data, batch_size, TIME_STEPS, name="TestInput")

# train_num_examples = train_input.shape[0]
# valid_num_examples = valid_input.shape[0]
# test_num_examples = test_input.shape[0]


print("The time distributed training data: " + str(train_input.input_data))
print("The similarly distributed targets: " + str(train_input.targets))

VOCAB_SIZE = 10000
EMBEDDING_SIZE = 100

# setup input and embedding
embedding_matrix = tf.get_variable('embedding_matrix', dtype=tf.float32, shape=[VOCAB_SIZE, EMBEDDING_SIZE], trainable=True)
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, train_input.input_data)
print("The output of the word embedding: " + str(word_embeddings))

LSTM_SIZE = 100 # number of units in the LSTM layer, this number taken from a "small" language model

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)

# Initial state of the LSTM memory.
initial_state = lstm_cell.zero_state(batch_size, tf.float32)
print("Initial state of the LSTM: " + str(initial_state))

outputs, state = rnn_block(lstm_cell, word_embeddings, initial_state)

logits = tf.layers.dense(outputs, VOCAB_SIZE)

LEARNING_RATE = 1e-4

loss = tf.contrib.seq2seq.sequence_loss(
    logits,
    train_input.targets,
    tf.ones([batch_size, TIME_STEPS], dtype=tf.float32),
    average_across_timesteps=True,
    average_across_batch=True)

optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
train_op = optimizer.minimize(loss)

with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    # start queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    print("Training begins")
    _, train_sequence_loss = session.run([train_op, loss])
    print('Train Sequence Loss: ' + str(train_sequence_loss))
    print("Training ends")

    test_sequence_loss = session.run(loss)
    print('Test Sequence Loss: ' + str(test_sequence_loss))

    # report mean validation loss
    valid_sequence_loss = session.run(loss)
    print('Valid Sequence Loss: ' + str(valid_sequence_loss))



# # retrieve some data to look at
# examples = session.run([train_input.input_data, train_input.targets])
# # we can run the train op as usual
# _ = session.run(train_op)

print("Example input data:\n" + str(examples[0][1]))
print("Example target:\n" + str(examples[1][1]))
