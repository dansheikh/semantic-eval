import numpy as np
import tensorflow as tf

import tensor_tools as tt
from tensor_tools import lazy_property


class RCRF():
    """Convenience class for Recurrent Neural Network combined with Conditional Random Field (RCRF).

    Args:
        rnn_size: Number of units in LSTM cell.
        num_layers: Number of layers in stacked LSTM.
        num_labels: Number of labels in target.
        num_words: Size of batches.
        num_feats: Number of features of input.
        optimizer_type: Type of optimizer function.
        eta: Learning rate.
        bi: Bidirectional flag.
        input_keep_prob: Probability of retaining input.
    """
    def __init__(self, rnn_size, num_layers, num_labels, num_words, num_feats, optimizer_type, eta, bi=False, input_keep_prob=None):
        self._rnn_size = rnn_size
        self._num_layers = num_layers
        self._num_labels = num_labels
        self._num_words = num_words
        self._num_feats = num_feats
        self._eta = eta
        self._bi = bi
        self._input_keep_prob = input_keep_prob
        self._dynamic_output = None
        self._dynamic_state = None
        self._logits = None
        self._log_likelihood = None
        self._trans_params = None

        if optimizer_type == 'adam':
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._eta)
        elif optimizer_type == 'rms':
            self._optimizer = tf.train.RMSPropOptimizer(learning_rate=self._eta)
        elif optimizer_type == 'sgd':
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._eta)

        self._lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size)

        if self._input_keep_prob is not None:
            self._lstm_cell = tf.contrib.rnn.DropoutWrapper(self._lstm_cell, input_keep_prob=self._input_keep_prob)

        self._lstm = tf.contrib.rnn.MultiRNNCell([self._lstm_cell] * self._num_layers, state_is_tuple=True)

        with tf.name_scope('inputs'):
            self._seq_lens = tf.placeholder(tf.int32, shape=(None), name='seq_lens')
            self._init_state = tf.placeholder(tf.float32, shape=(self._num_layers, 2, None, self._rnn_size), name='init_state')
            self._x = tf.placeholder(tf.float32, shape=(None, self._num_words, self._num_feats), name='x')  # None is fill-in for dynamically padded sentences.
            self._y = tf.placeholder(tf.int32, shape=(None, self._num_words), name='y')  # None is fill-in for dynamically padded labels.

        # TensorFlow 'functions'.
        self.dynamic_run
        self.fully_connect
        self.crf_log_likelihood
        self.loss
        self.optimize

    @property
    def rnn_size(self):
        return self._rnn_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def seq_lens(self):
        return self._seq_lens

    @seq_lens.setter
    def seq_lens(self, x):
        self._seq_lens = x

    @property
    def init_state(self):
        return self._init_state

    @init_state.setter
    def init_state(self, x):
        self._init_state = x

    @property
    def dynamic_output(self):
        return self._dynamic_output

    @dynamic_output.setter
    def dynamic_output(self, x):
        self._dynamic_output = x

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
    def log_likelihood(self):
        return self._log_likelihood

    @log_likelihood.setter
    def log_likelihood(self, x):
        self._log_likelihood = x

    @property
    def trans_params(self):
        return self._trans_params

    @trans_params.setter
    def trans_params(self, x):
        self._trans_params = x

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
        # Note: state re-construction currently does NOT account for forward/backward states.
        # Initial state: [num_layers x 2 x batch_size x rnn_size]
        # Bi-directional forward-feed state: [(num_layers * 2) x 2 x batch_size x rnn_size]
        unstacked_state = tf.unstack(self._init_state)
        state_tuple = tuple([tf.contrib.rnn.LSTMStateTuple(unstacked_state[idx][0], unstacked_state[idx][1]) for idx in range(self._num_layers)])
        output = None
        state = None

        if self._bi:
            (outputs, states) = tf.nn.bidirectional_dynamic_rnn(self._lstm, self._lstm, self._x, sequence_length=self._seq_lens, initial_state_fw=state_tuple, initial_state_bw=state_tuple, dtype=tf.float32)
            output = tf.concat(values=outputs, axis=2)
            state = tf.concat(values=states, axis=0)
        else:
            (output, state) = tf.nn.dynamic_rnn(self._lstm, self._x, sequence_length=self._seq_lens, initial_state=state_tuple, dtype=tf.float32)

        self._dynamic_output = output
        self._dynamic_state = state

        return (self._dynamic_output, self._dynamic_state)

    @lazy_property
    def fully_connect(self):
        if self._bi:
            output = tf.reshape(self._dynamic_output, [-1, (self._rnn_size * 2)])
        else:
            output = tf.reshape(self._dynamic_output, [-1, self._rnn_size])

        tmp_logits = tf.contrib.layers.fully_connected(inputs=output, num_outputs=self._num_labels)
        self._logits = tf.reshape(tmp_logits, [-1, self._num_words, self._num_labels])
        return self._logits

    @lazy_property
    def crf_log_likelihood(self):
        (self._log_likelihood, self._trans_params) = tf.contrib.crf.crf_log_likelihood(self.fully_connect, self._y, self._seq_lens)
        return (self._log_likelihood, self._trans_params)

    @lazy_property
    def loss(self):
        log_likelihood, trans_param = self.crf_log_likelihood
        return tf.reduce_mean(-log_likelihood)

    @lazy_property
    def optimize(self):
        objective = self._optimizer.minimize(self.loss)

        return objective
