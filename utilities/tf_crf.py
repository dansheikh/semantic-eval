import tensorflow as tf
import tensor_tools as tt

KEYPHRASES = 3
EMBED_SIZE = 100

x = tf.placeholder(tf.float32, [None, EMBED_SIZE])
W = tt.weight_variable([EMBED_SIZE, KEYPHRASES])
b = tt.bias_variable([KEYPHRASES])
y = tf.placeholder(tf.float32, [None, KEYPHRASES])
