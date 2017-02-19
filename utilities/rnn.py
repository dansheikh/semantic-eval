import tensorflow as tf
import tensor_tools as tt


class MultiRNNLSTM():
    """Convenience class for Stacked LSTMs.

    Args:
        rnn_size: Size of RNN.
        num_layers: Number of layers in stacked LSTM.
        num_labels: Number of labels in targt.
        batch_size: Size of batches.
        features: Number of features of input.
        eta: Learning rate.
    """
    def __init__(self, rnn_size, num_layers, num_labels, batch_size, features, eta):
        self._rnn_size = rnn_size
        self._num_layers = num_layers
        self._num_labels = num_labels
        self._batch_size = batch_size
        self._features = features
        self._eta = eta
        self._lstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        self._lstm = tf.nn.rnn_cell.MultiRNNCell([self._lstm] * self._num_layers)
        self._init_state = self._lstm.zero_state(self._batch_size, tf.float32)

        with tf.variable_scope('lstm_vars'):
            self._W = tt.weight_variable([self._rnn_size, self._num_labels])
            self._b = tt.bias_variable([self._num_labels])

        self._x = tf.placeholder(tf.float32, [self._batch_size, self._features])
        self._y = tf.placeholder(tf.int32, [self._batch_size, self._num_labels])

        output, state = tf.nn.dynamic_rnn(self._lstm, tf.reshape(self._x, [1, self._batch_size, self._features]), dtype=tf.float32)
        self._h = tf.reshape(output, [-1, self._rnn_size])
        self._logits = tf.matmul(self._h, self._W) + self._b
        self._y_hat = tf.nn.softmax(self._logits)
        self._expected = tf.argmax(self._y, 1)
        self._predicted = tf.argmax(self._y_hat, 1)
        self._correct = tf.equal(self._expected, self._predicted)
        self._accuracy = tf.reduce_mean(tf.cast(self._correct, tf.float32))

        self._cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self._y_hat, self._y))
        self._optimize = tf.train.GradientDescentOptimizer(self._eta).minimize(self._cross_entropy)

    @property
    def rnn_size(self):
        return self._rnn_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def init_state(self):
        return self._init_state

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

    @property
    def y_hat(self):
        return self._y_hat

    @property
    def optimize(self):
        return self._optimize

    @property
    def expected(self):
        return self._expected

    @property
    def predicted(self):
        return self._predicted

    @property
    def accuracy(self):
        return self._accuracy
