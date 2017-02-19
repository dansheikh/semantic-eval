import collections
import numpy as np
import tensorflow as tf
import gensim


def weight_variable(shape):
    """Creates truncated normally distributed tensorflow variable.

    Args:
        shape: Variable shape.

    Returns:
        A tensorflow variable matching requested shape and with truncated normally distributed values.
    """
    return tf.get_variable("weight", initializer=tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    """Creates a tensorflow constant.

    Args:
        shape: Constant shape.

    Returns:
        A tensorflow constant matching requested shape with all values set to 0.1.
    """
    return tf.get_variable("bias", initializer=tf.constant(0.1, shape=shape))


def preprocess(word2vec_path, lbls_path, sep='\t'):
    """Use word2vec model to create vectorized input data and extract labels from tagged file.

    Args:
        word2vec_path: Path to trained word2vec model.
        lbl_path: Path to CoNLL formatted file.
        sep: CoNLL file content separator.

    Returns:
        Word2Vec embedding model, vectorized input data and a corresponding sequence of labels.
    """
    embedding = gensim.models.Word2Vec.load(word2vec_path)
    word_vecs = list()
    labels = list()

    with open(lbls_path, mode='r') as file:

        for idx, line in enumerate(file):
            content = line.rstrip().split(sep)
            if content[0].strip():
                word_vecs.append(embedding[content[1]])
                labels.append(content[0])

    return embedding, word_vecs, labels


def one_hot(labels):
    """Encoded labels into one-hot format.

    Args:
        labels: A sequence of labels.

    Returns:
        A nested array of one-hot encoded labels and accompanying label dictionary.
    """
    label_set = set(labels)
    label_dict = dict()
    encoding = np.zeros((len(labels), len(label_set)))

    for i, lbl in enumerate(label_set):
        label_dict[lbl] = i

    label_dict = collections.OrderedDict(sorted(label_dict.items(), key=lambda e: e[1]))

    for j, label in enumerate(labels):
        encoding[j, label_dict[label]] = 1

    return encoding, label_dict


def dict_to_list(d):
    """Converts dictionary to list using dictionary values as sorting key."""
    tmp = [None] * len(d)
    for k, v in d.items():
        tmp[v] = k

    return tmp


def batch_calc(num, origin):
    divisor = origin
    delta = 1

    while True:
        if num % divisor == 0:
            break

        if num % (divisor + delta) == 0:
            divisor += delta
            break
        elif num % (divisor - delta) == 0:
            divisor -= delta
            break
        else:
            delta += 1

    section_size = num // divisor

    return divisor, section_size
