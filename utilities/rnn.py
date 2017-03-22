import numpy as np
import tensorflow as tf

import tensor_tools as tt
from tensor_tools import lazy_property


class MultiRNNLSTM():
    """Convenience class for Stacked LSTMs.

    Args:
        rnn_size: Number of units in LSTM cell.
        num_layers: Number of layers in stacked LSTM.
        num_labels: Number of labels in target.
        batch_size: Size of batches.
        features: Number of features of input.
        eta: Learning rate.
        input_keep_prob: Probability of retaining input.
    """
    def __init__(self, rnn_size, num_layers, num_labels, batch_size, feature_size, eta, input_keep_prob=None):
        self._rnn_size = rnn_size
        self._num_layers = num_layers
        self._num_labels = num_labels
        self._batch_size = batch_size
        self._feature_size = feature_size
        self._eta = eta
        self._input_keep_prob = input_keep_prob
        self._dynamic_state = None
        self._logits = None

        self._lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size)
        if self._input_keep_prob is not None:
            self._lstm_cell = tf.contrib.rnn.DropoutWrapper(self._lstm_cell, input_keep_prob=self._input_keep_prob)
        self._lstm = tf.contrib.rnn.MultiRNNCell([self._lstm_cell] * self._num_layers, state_is_tuple=True)

        with tf.name_scope('inputs'):
            self._seq_len = tf.placeholder(tf.float32, shape=(self._batch_size), name='seq_len')
            self._init_state = tf.placeholder(tf.float32, shape=(self._num_layers, 2, self._batch_size, self._rnn_size), name='init_state')
            self._x = tf.placeholder(tf.float32, shape=(self._batch_size, None, self._feature_size), name='x')  # None is fill-in for dynamically padded sentences.
            self._y = tf.placeholder(tf.int32, shape=(self._batch_size, None, self._num_labels), name='y')  # None is fill-in for dynamically padded labels.

        with tf.variable_scope('lstm_vars'):
            self._lstm_W = tt.weight_variable([self._rnn_size, 100], 'lstm_weight')
            self._lstm_b = tt.bias_variable([100], 'lstm_bias')
            self._W = tt.weight_variable([100, self._num_labels], 'weight')
            self._b = tt.bias_variable([self._num_labels], 'bias')

        # TensorFlow 'functions'.
        self.dynamic_run
        self.y_hat
        self.expect
        self.predict
        self.cross_entropy
        self.optimize
        self.accuracy

    @property
    def rnn_size(self):
        return self._rnn_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def seq_len(self):
        return self._seq_len

    @seq_len.setter
    def seq_len(self, x):
        self._seq_len = x

    @property
    def init_state(self):
        return self._init_state

    @init_state.setter
    def init_state(self, x):
        self._init_state = x

    @property
    def dynamic_state(self):
        return self._dynamic_state

    @dynamic_state.setter
    def dynamic_state(self, x):
        self._dynamic_state = x

    @property
    def logits(self):
        return self._logits

    @logits.setter
    def logits(self, x):
        self._logits = x

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y

    @lazy_property
    def dynamic_run(self):
        unstacked_state = tf.unstack(self._init_state)
        state_tuple = tuple([tf.contrib.rnn.LSTMStateTuple(unstacked_state[idx][0], unstacked_state[idx][1]) for idx in range(self._num_layers)])
        output, state = tf.nn.dynamic_rnn(self._lstm, self._x, sequence_length=self._seq_len, initial_state=state_tuple, dtype=tf.float32)

        return (output, state)

    @lazy_property
    def y_hat(self):
        (output, self._dynamic_state) = self.dynamic_run

        # output = tf.transpose(output, [1, 0, 2])  # Transpose to: [features x batch_size x rnn_size]
        # last_output = tf.gather(output, int(output.get_shape()[0] - 1))  # Gather: [batch_size x rnn_size]
        #
        # lstm_logits = tf.matmul(last_output, self._lstm_W) + self._lstm_b
        # relu_logits = tf.nn.relu(lstm_logits)
        # logits = tf.matmul(relu_logits, self._W) + self._b
        #
        output = tf.reshape(output, [-1, self._rnn_size])
        lstm_logits = tf.matmul(output, self._lstm_W) + self._lstm_b
        relu_logits = tf.nn.relu(lstm_logits)
        self._logits = tf.matmul(relu_logits, self._W) + self._b

        return tf.nn.softmax(self._logits)

    @lazy_property
    def expect(self):
        y = tf.reshape(self.y, [-1, self._num_labels])
        return tf.argmax(y, 1)

    @lazy_property
    def predict(self):
        return tf.argmax(self.y_hat, 1)

    @lazy_property
    def cross_entropy(self):
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.expect, logits=self.logits))
        tf.summary.scalar('cross_entropy', cross_entropy)

        return cross_entropy

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self._eta)
        objective = optimizer.minimize(self.cross_entropy)

        return objective

    @lazy_property
    def accuracy(self):
        correct = tf.equal(self.expect, self.predict)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        return accuracy


class MultiRNNGRU():
    """Convenience class for Stacked GRUs.

    Args:
        rnn_size: Number of units in GRU cell.
        num_layers: Number of layers in stacked GRU.
        num_labels: Number of labels in target.
        batch_size: Size of batches.
        features: Number of features of input.
        eta: Learning rate.
        input_keep_prob: Probability of retaining input.
    """
    def __init__(self, rnn_size, num_layers, num_labels, batch_size, mini_batch_size, mini_steps, features, eta, input_keep_prob=None):
        self._rnn_size = rnn_size
        self._num_layers = num_layers
        self._num_labels = num_labels
        self._batch_size = batch_size
        self._features = features
        self._mini_batch_size = mini_batch_size
        self._mini_steps = mini_steps
        self._eta = eta
        self._input_keep_prob = input_keep_prob
        self._dynamic_state = None

        self._gru_cell = tf.contrib.rnn.GRUCell(rnn_size)
        if self._input_keep_prob is not None:
            self._gru_cell = tf.contrib.rnn.DropoutWrapper(self._gru_cell, input_keep_prob=self._input_keep_prob)
        self._gru = tf.contrib.rnn.MultiRNNCell([self._gru_cell] * self._num_layers, state_is_tuple=True)

        with tf.name_scope('inputs'):
            self._init_state = tf.placeholder(tf.float32, shape=(self._num_layers, self._batch_size, self._rnn_size), name='init_state')
            self._dynamic_output = tf.placeholder(tf.float32, shape=(self._batch_size, self._mini_steps, self._rnn_size), name='dynamic_output')
            self._mini_input = tf.placeholder(tf.float32, shape=(self._batch_size, self._mini_steps), name='mini_inputs')
            self._x = tf.placeholder(tf.float32, shape=(self._batch_size, self._features), name='x')
            self._y = tf.placeholder(tf.int32, shape=(self._batch_size, self._num_labels), name='y')

        with tf.variable_scope('gru_vars'):
            self._gru_W = tt.weight_variable([self._rnn_size, 100], 'gru_weight')
            self._gru_b = tt.bias_variable([100], 'gru_bias')
            self._W = tt.weight_variable([100, self._num_labels], 'weight')
            self._b = tt.bias_variable([self._num_labels], 'bias')

        # TensorFlow 'functions'.
        self.dynamic_run
        self.y_hat
        self.expect
        self.predict
        self.cross_entropy
        self.optimize
        self.accuracy

    @property
    def rnn_size(self):
        return self._rnn_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def dynamic_output(self):
        return self._dynamic_output

    @dynamic_output.setter
    def dynamic_output(self, x):
        self._dynamic_output = x

    @property
    def init_state(self):
        return self._init_state

    @init_state.setter
    def state(self, x):
        self._init_state = x

    @property
    def dynamic_state(self):
        return self._dynamic_state

    @dynamic_state.setter
    def dynamic_state(self, x):
        self._dynamic_state = x

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def mini_input(self):
        return self._mini_input

    @mini_input.setter
    def mini_input(self, x):
        self._mini_input = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y

    @lazy_property
    def dynamic_run(self):
        data = tf.reshape(self._mini_input, [self._batch_size, -1, 1])
        unstacked_state = tf.unstack(self._init_state)
        state_tuple = tuple([(unstacked_state[idx]) for idx in range(self._num_layers)])

        output, state = tf.nn.dynamic_rnn(self._gru, data, initial_state=state_tuple, dtype=tf.float32)

        return output, state

    @lazy_property
    def y_hat(self):
        (output, self._dynamic_state) = self.dynamic_run

        # output = tf.transpose(output, [1, 0, 2])  # Transpose to: [features x batch_size x rnn_size]
        # last_output = tf.gather(output, int(output.get_shape()[0] - 1))  # Gather: [batch_size x rnn_size]
        #
        # lstm_logits = tf.matmul(last_output, self._lstm_W) + self._lstm_b
        # relu_logits = tf.nn.relu(lstm_logits)
        # logits = tf.matmul(relu_logits, self._W) + self._b
        #
        output = tf.reshape(output, [-1, self._rnn_size])
        lstm_logits = tf.matmul(output, self._lstm_W) + self._lstm_b
        relu_logits = tf.nn.relu(lstm_logits)
        logits = tf.matmul(relu_logits, self._W) + self._b

        return tf.nn.softmax(logits)

    @lazy_property
    def expect(self):
        return tf.argmax(self.y, 1)

    @lazy_property
    def predict(self):
        return tf.argmax(self.y_hat, 1)

    @lazy_property
    def cross_entropy(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat))
        tf.summary.scalar('cross_entropy', cross_entropy)

        return cross_entropy

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self._eta)
        objective = optimizer.minimize(self.cross_entropy)

        return objective

    @lazy_property
    def accuracy(self):
        correct = tf.equal(self.expect, self.predict)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        return accuracy
