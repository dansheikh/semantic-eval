#!/usr/bin/env python

import argparse
import os

import numpy as np
import tensorflow as tf

import tensor_tools as tt


def _learn(args):
    label_dict = None
    if args.nontyped:
        label_dict = {'O': 0, 'Keyphrase': 1}
    else:
        label_dict = {'O': 0, 'Material': 1, 'Process': 2, 'Task': 3}
    num_lbls = len(label_dict)
    (sequences, labels) = tt.preprocess(args.word2vec_path, args.labels_path)  # Prepare zipped sequences of vectored words and labels.
    scalar_labels = tt.numeric_labels(labels, label_dict)
    num_seqs = len(sequences)
    num_feats = np.shape(sequences[0])[1]
    seq_lens = [len(sequence) for sequence in sequences]
    seq_lens = np.array(seq_lens)

    # Pad sentences and labels.
    pad_seqs, pad_lbls = tt.pad_data(sequences, scalar_labels, label_dict)
    num_words = np.shape(pad_seqs)[1]

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=(None, num_words, num_feats), name='sentences')
        x_lens = tf.placeholder(tf.int32, shape=(None), name='sentence_lens')
        gold_lbls = tf.placeholder(tf.int32, shape=(None, num_words), name='gold_lbls')

    with tf.variable_scope('crf'):
        weights = tt.weight_variable([num_feats, num_lbls], 'weights')

    x_matrix = tf.reshape(x, [-1, num_feats])
    unary_scores_matrix = tf.matmul(x_matrix, weights)
    unary_scores = tf.reshape(unary_scores_matrix, [-1, num_words, num_lbls])

    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(unary_scores, gold_lbls, x_lens)

    loss = tf.reduce_mean(-log_likelihood)
    optimizer = None

    if args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(args.alpha)
    elif args.optimizer == 'rms':
        optimizer = tf.train.RMSPropOptimizer(args.alpha)
    elif args.optimzier == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(args.alpha)

    objective = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        learning = True
        epoch = 0
        plateau_cnt = 0
        num_batches = num_seqs // args.target_batch_size
        epoch_accuracy = []
        epoch_loss = []

        while learning:
            correct_lbls = 0
            totals_lbls = 0

            if (epoch + 1) % 10 == 0:
                epoch_accuracy.append([])
                epoch_loss.append([])

            for batch_idx in np.arange(num_batches):
                seq_batch = None
                lbl_batch = None
                batch_lens = None

                if batch_idx < num_batches:
                    seq_batch = pad_seqs[(batch_idx * args.target_batch_size): ((batch_idx + 1) * args.target_batch_size)]
                    lbl_batch = pad_lbls[(batch_idx * args.target_batch_size): ((batch_idx + 1) * args.target_batch_size)]
                    batch_lens = seq_lens[(batch_idx * args.target_batch_size): ((batch_idx + 1) * args.target_batch_size)]
                else:
                    seq_batch = pad_seqs[(batch_idx * args.target_batch_size):]
                    lbl_batch = pad_lbls[(batch_idx * args.target_batch_size):]
                    batch_lens = seq_lens[(batch_idx * args.target_batch_size):]

                feed_dict = {x: seq_batch, x_lens: batch_lens, gold_lbls: lbl_batch}
                sess_unary_scores, sess_trans_params, sess_loss, _ = sess.run([unary_scores, trans_params, loss, objective], feed_dict=feed_dict)

                if (epoch + 1) % 10 == 0:
                   for (sess_score, sess_lbl, sess_seq_len) in zip(sess_unary_scores, lbl_batch, seq_lens):
                        sess_score = sess_score[:sess_seq_len]
                        sess_lbl = sess_lbl[:sess_seq_len]
                        viterbi_seq, _ = tf.contrib.crf.viterbi_decode(sess_score, sess_trans_params)

                        correct_lbls += np.sum(np.equal(viterbi_seq, sess_lbl))
                        totals_lbls += sess_seq_len

                    accuracy = 100 * correct_lbls / float(totals_lbls)
                    epoch_accuracy[-1].append(accuracy)
                    epoch_loss[-1].append(sess_loss)

            if len(epoch_loss) > 2:
                previous_loss = np.mean(epoch_loss[-2])
                current_loss = np.mean(epoch_loss[-1])
                loss_diff = previous_loss - current_loss
                print("Epoch [{}] Accuracy {:.2f} | Loss: {:.5f} | Loss Diff: {:.5f}".format(epoch, np.mean(epoch_accuracy[-1]), current_loss, loss_diff))

                if abs(loss_diff) < args.epsilon:
                    plateau_cnt += 1

            if plateau_cnt >= 3:
                learning = False

            if epoch >= args.kill_zone:
                learning = False
            epoch += 1

        path = os.path.join(args.save_path, '')
        filename = os.path.splitext(os.path.basename(__file__))[0]
        checkpoint = "{path}{filename}.ckpt".format(path=path, filename=filename)
        print("Saving model variables to: {checkpoint}".format(checkpoint=checkpoint))
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)


def _eval(args):
    label_dict = None
    if args.nontyped:
        label_dict = {'O': 0, 'Keyphrase': 1}
    else:
        label_dict = {'O': 0, 'Material': 1, 'Process': 2, 'Task': 3}

    label_list = [None for _ in range(len(label_dict))]
    for key, val in label_dict.items():
        label_list[val] = key

    num_lbls = len(label_dict)
    (sequences, labels) = tt.preprocess(args.word2vec_path, args.labels_path)  # Prepare zipped sequences of vectored words and labels.
    scalar_labels = tt.numeric_labels(labels, label_dict)
    num_seqs = len(sequences)
    num_feats = np.shape(sequences[0])[1]
    seq_lens = [len(sequence) for sequence in sequences]

    # Pad sentences and labels.
    pad_seqs, pad_lbls = tt.pad_data(sequences, scalar_labels, label_dict)
    num_words = np.shape(pad_seqs)[1]

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=(num_seqs, num_words, num_feats), name='sentences')
        x_lens = tf.placeholder(tf.int32, shape=(None), name='sentence_lens')
        gold_lbls = tf.placeholder(tf.int32, shape=(num_seqs, None), name='gold_lbls')

    with tf.variable_scope('crf'):
        weights = tt.weight_variable([num_feats, num_lbls], 'weights')

    x_matrix = tf.reshape(x, [-1, num_feats])
    unary_scores_matrix = tf.matmul(x_matrix, weights)
    unary_scores = tf.reshape(unary_scores_matrix, [num_seqs, num_words, num_lbls])

    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(unary_scores, gold_lbls, x_lens)

    loss = tf.reduce_mean(-log_likelihood)
    objective = tf.train.GradientDescentOptimizer(args.alpha).minimize(loss)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    path = os.path.join(args.load_path, '')
    filename = os.path.splitext(os.path.basename(__file__))[0]
    checkpoint = "{path}{filename}.ckpt".format(path=path, filename=filename)

    with tf.Session() as sess:
        sess.run(init)
        print("Loading {checkpoint}...".format(checkpoint=checkpoint))
        saver.restore(sess, checkpoint)  # Load trained weights and biases.
        print("Loaded {checkpoint}.".format(checkpoint=checkpoint))

        feed_dict = {x: pad_seqs, x_lens: seq_lens, gold_lbls: pad_lbls}
        sess_unary_scores, sess_trans_params, sess_loss, _ = sess.run([unary_scores, trans_params, loss, objective], feed_dict=feed_dict)

        path = os.path.join(args.conll_path, '')
        filename = os.path.splitext(os.path.basename(__file__))[0]
        conll_file = "{path}{filename}.log".format(path=path, filename=filename)

        # Write expected and predicted values to file in CoNLL format.
        with open(conll_file, mode='a+') as conll:
            for (sess_score, sess_lbl, sess_seq_len) in zip(sess_unary_scores, pad_lbls, seq_lens):
                sess_score = sess_score[:sess_seq_len]
                sess_lbl = sess_lbl[:sess_seq_len]
                viterbi_seq, _ = tf.contrib.crf.viterbi_decode(sess_score, sess_trans_params)
                for expect, predict in zip(sess_lbl, viterbi_seq):
                    line = "{e}\t{p}\n".format(e=label_list[expect], p=label_list[predict])
                    conll.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word2Vec_CRF')
    parser.add_argument('-a', '--alpha', action='store', default=0.001, type=float)
    parser.add_argument('-b', '--board_path', action='store', default='logs/', type=str)
    parser.add_argument('-c', '--conll_path', action='store')
    parser.add_argument('-e', '--epsilon', action='store', default=0.001, type=float)
    parser.add_argument('-k', '--kill_zone', action='store', default=1000000, type=int)
    parser.add_argument('-l', '--load_path', action='store')
    parser.add_argument('-m', '--mode', action='store', required=True, choices=['learn', 'eval'])
    parser.add_argument('--nontyped', action='store_true')
    parser.add_argument('-o', '--optimizer', action='store', choices=['adam', 'rms', 'sgd'], default='adam')
    parser.add_argument('-p', '--peek', action='store', default='', type=str)
    parser.add_argument('-s', '--save_path', action='store')
    parser.add_argument('-t', '--target_batch_size', action='store', default=25, type=int)
    parser.add_argument('word2vec_path')
    parser.add_argument('labels_path')
    args = parser.parse_args()

    if args.mode == 'learn':
        if args.save_path is None:
            parser.error('Mode "learn" requires save path.')

        print('Learning...')
        _learn(args)
        print('Learning complete.')
    else:
        if args.load_path is None or args.conll_path is None:
            parser.error('Mode "eval" requires load path and CoNLL path.')

        print('Evaluating...')
        _eval(args)
        print('Evaluation complete.')
