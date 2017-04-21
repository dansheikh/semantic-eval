import csv
import functools
import os

import gensim
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def lazy_property(func):
    """Decorator lazy properties."""
    attr = '_cache_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr):
            setattr(self, attr, func(self))
        else:
            return getattr(self, attr)

    return wrapper


def csv_log(file, headers, inputs):
    header_titles = ['' for w in np.arange(len(headers))]

    for k, v in headers.items():
        header_titles[v] = k

    with open(file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, dialect='excel', delimiter=',', quotechar='"')
        header_row = ['Epoch Id', 'Batch Id', 'Step Id']
        header_row.extend(header_titles)
        writer.writerow(header_row)

        for e, epoch in enumerate(inputs):
            for b, batches in enumerate(epoch):
                for s, batch in enumerate(batches):
                    data = [e, b, s]
                    data.extend(batch)
                    writer.writerow(data)


def diagram(loss, accuracy):
    """Creates loss and accuracy line graphs."""
    if type(loss) is not np.ndarray:
        loss = np.array(loss)
    if type(accuracy) is not np.ndarray:
        accuracy = np.array(accuracy)

    if len(np.shape(loss)) > 1:
        loss = np.mean(loss, axis=0)

    if len(np.shape(accuracy)) > 1:
        accuracy = np.mean(accuracy, axis=0)

    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax1.set_title('Loss')
    ax2.set_title('Accuracy')

    sampled_loss = loss[0::10]
    sampled_accuracy = accuracy[0::10]

    ax1.plot(np.arange(np.shape(sampled_loss)[0]), sampled_loss, linewidth=1.5, color='b')
    ax2.plot(np.arange(np.shape(sampled_accuracy)[0]), sampled_accuracy, linewidth=1.5, color='c')

    plt.show()


def weight_variable(shape, name):
    """Creates truncated normally distributed tensorflow variable.

    Args:
        shape: Variable shape.

    Returns:
        A tensorflow variable matching requested shape and with truncated normally distributed values.
    """
    return tf.get_variable(name, initializer=tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape, name):
    """Creates a tensorflow constant.

    Args:
        shape: Constant shape.

    Returns:
        A tensorflow constant matching requested shape with all values set to 0.1.
    """
    return tf.get_variable(name, initializer=tf.constant(0.1, shape=shape))


def preprocess(word2vec_path, data_path, sep='\t'):
    """Use word2vec model to create vectorized input data and extract labels from tagged file.

    Args:
        word2vec_path: Path to trained word2vec model.
        data_path: Path to CoNLL formatted file.
        sep: CoNLL file content separator.

    Returns:
        Vectorized input sequences and corresponding sequence of labels.
    """
    word2vec_model = gensim.models.Word2Vec.load(word2vec_path)
    embeds = word2vec_model.wv.syn0

    # Set constants.
    BEG = '__BOS__'
    END = '__EOS__'
    UNK = np.mean(embeds, axis=0)

    # Containers.
    sequences = list()
    labels = list()

    with open(data_path, mode='r', encoding='utf-8') as file:
        input_seq = None
        lbl_seq = None

        for idx, line in enumerate(file):
            line = line.rstrip(os.linesep).split(sep)

            if len(line) == 3 and line[2] == BEG:
                # Start new sequence.
                input_seq = list()
                lbl_seq = list()

                if line[1] in word2vec_model:
                    input_seq.append(word2vec_model[line[1]])
                    lbl_seq.append(line[0])
                else:
                    input_seq.append(UNK)
                    lbl_seq.append(line[0])

            elif len(line) == 3 and line[2] == END:
                if line[1] in word2vec_model:
                    input_seq.append(word2vec_model[line[1]])
                    lbl_seq.append(line[0])

                else:
                    input_seq.append(UNK)
                    lbl_seq.append(line[0])

                sequences.append(input_seq)
                labels.append(lbl_seq)

            elif len(line) == 2:
                if line[1] in word2vec_model:
                    input_seq.append(word2vec_model[line[1]])
                    lbl_seq.append(line[0])
                else:
                    input_seq.append(UNK)
                    lbl_seq.append(line[0])

    return (sequences, labels)


def package_batch(chosen_seqs, sequences, labels, label_dict):
    """"Package sequences and labels into consumable training batches.

    Args:
        chosen_seqs: List of chosen sequences.
        sequences: List of sequences.
        labels: List of labels associated with provided sequences.

    Returns:
        Vector of sequence lengths for each chosen sequence and a tuple of sequences and labels.
    """
    seq_len = np.zeros(len(chosen_seqs))
    lbl_cnt = len(label_dict)
    out_idx = label_dict['O']
    OUT = np.zeros((1, lbl_cnt))
    OUT[0][out_idx] = 1
    PAD = np.zeros((1, np.shape(sequences[0])[1]))  # Determine feature size from provided sequence element.
    packed_seqs = []
    packed_lbls = []

    # Collect sequence lengths.
    for idx, chosen_seq in enumerate(chosen_seqs):
        seq_len[idx] = len(sequences[chosen_seq])

    max_len = np.amax(seq_len)

    # Pad sequences to uniform length.
    for idx, chosen_seq in enumerate(chosen_seqs):
        seq = sequences[chosen_seq]
        lbl = labels[chosen_seq]
        diff = max_len - len(seq)

        for n in np.arange(diff):
            seq = np.append(seq, PAD, axis=0)
            lbl = np.append(lbl, OUT, axis=0)

        packed_seqs.append(seq)
        packed_lbls.append(lbl)

    return (seq_len, zip(packed_seqs, packed_lbls))


def pad_data(seqs, lbls, lbl_dict):
    """"Pad sequences and labels to uniform feature size.

    Args:
        seqs: List of sequences.
        lbls: List of labels associated with provided sequences.

    Returns:
        A tuple of sequences and labels.
    """
    out_idx = lbl_dict['O']
    PAD = np.zeros(np.shape(seqs[0])[1])  # Determine feature size from provided sequence element.
    seq_cnt = len(seqs)
    max_len = 0

    # Determine max sequence length.
    for i, seq in enumerate(seqs):
        seq_len = len(seq)
        if seq_len > max_len:
            max_len = seq_len

    for j in np.arange(seq_cnt):
        seq_len = len(seqs[j])
        lbl_len = len(lbls[j])

        seq_len_diff = max_len - seq_len
        lbl_len_diff = max_len - lbl_len

        seq_addon = [PAD for _ in np.arange(seq_len_diff)]
        lbl_addon = [out_idx for _ in np.arange(lbl_len_diff)]

        seqs[j].extend(seq_addon)
        lbls[j].extend(lbl_addon)

    return (seqs, lbls)


def one_hot(sequences, labels):
    """Encode labels into one-hot format.

    Args:
        sequences: A sequence of values to encode.
        labels: A dictionary of labels

    Returns:
        A nested array of one-hot encoded labels.
    """
    lbl_cnt = len(labels)
    encoding = []

    for i, inputs in enumerate(sequences):
        inputs_cnt = len(inputs)
        one_hot_seq = np.zeros((inputs_cnt, lbl_cnt))

        for j, input in enumerate(inputs):
            one_hot_seq[j, labels[input]] = 1

        encoding.append(one_hot_seq)

    return encoding


def numeric_labels(sequences, labels):
    """Encode labels into numeric format.

    Args:
        sequences: A sequence of values to encode.
        labels: A dictionary of labels

    Returns:
        A nested array of numeric labels.
    """
    encoding = []

    for i, inputs in enumerate(sequences):
        lbl_seq = []

        for j, input in enumerate(inputs):
            lbl_seq.append(labels[input])

        encoding.append(lbl_seq)

    return encoding


def epoch_mean(seq):
    def _mean(item):
        if len(item) > 0:
            return sum(item) / float(len(item))
        else:
            return None

    return [_mean(item) for i, item in enumerate(seq)]


def random_choices(num, picks):
    """"Return a random sampling from 0 to supplied number."""
    rounds = num // picks
    choices = [np.random.choice(num, picks) for _ in np.arange(rounds)]
    return (rounds, choices)


def batch_calc(num, target):
    divisor = target
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

    return (divisor, section_size)


def binarize(images, threshold=0.1):
    """Converts image to binary representation."""
    return (threshold < images).astype('float32')


def clear_logs(path):
    """Clear logs directory"""
    files = os.listdir(path)

    for file in files:
        full_path = os.path.join(path, file)
        if os.path.isfile(full_path):
            os.remove(full_path)
