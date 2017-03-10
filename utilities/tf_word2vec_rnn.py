#!/usr/bin/env python
import argparse
import os

import numpy as np
import tensorflow as tf

import rnn
import tensor_tools as tt


def _learn(args):
    label_dict = {'O': 0, 'Material': 1, 'Process': 2, 'Task': 3}
    num_labels = len(label_dict)
    (sequences, labels) = tt.preprocess(args.word2vec_path, args.labels_path)  # Prepare zipped sequences of vectored words and labels.
    num_inputs = len(sequences)  # Determine number of training examples.
    feature_size = len(sequences[0][0])  # Determine word vector size.
    one_hot_labels = tt.one_hot(labels, label_dict)

    # Calculate batches.
    rounds, choices = tt.random_choices(num_inputs, args.target_batch_size)
    batch_size = args.target_batch_size
    _, iter_wait = tt.batch_calc(rounds, args.target_batch_size)

    with tf.variable_scope('lstm_model'):
        model = rnn.MultiRNNLSTM(args.rnn_size, args.depth, num_labels, batch_size, feature_size, args.alpha, args.input_keep_prob)

    saver = tf.train.Saver()
    sum_merge = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        logpath = os.path.join(args.board_path, 'train', '')
        tt.clear_logs(logpath)
        writer = tf.summary.FileWriter(args.board_path + 'train', graph=sess.graph)
        sess.run(init)

        losses = np.zeros((args.epochs, rounds))
        accuracies = np.zeros((args.epochs, rounds))

        # Train model.
        for epoch in np.arange(args.epochs):
            steps = np.arange(rounds)  # Each step is a batch round.

            state = np.zeros((args.depth, 2, batch_size, args.rnn_size))  # Set empty initial state for each batch.
            for step in steps:
                chosen_seqs = choices[step]
                (seq_len, seq_lbl_zip) = tt.package_batch(chosen_seqs, sequences, one_hot_labels, label_dict)
                batch_x, batch_y = zip(*seq_lbl_zip)

                feed_dict = {model.x: batch_x, model.y: batch_y, model.seq_len: seq_len, model.init_state: state}

                _, train_cross_entropy, train_accuracy, state, summary = sess.run([model.optimize, model.cross_entropy, model.accuracy, model.dynamic_state, sum_merge], feed_dict=feed_dict)
                losses[epoch, step] = train_cross_entropy
                accuracies[epoch, step] = train_accuracy

                if (step + 1) % iter_wait == 0:
                    print("[Epoch {epoch}, Step {step:0>4d}] Loss: {loss:.5f} | Accuracy: {accuracy:.5f}".format(epoch=epoch, step=step, loss=train_cross_entropy, accuracy=train_accuracy))

                writer.add_summary(summary, step)

        path = os.path.join(args.save_path, '')
        filename = os.path.split(os.path.basename(__file__))[0]
        checkpoint = "{path}{filename}.ckpt".format(path=path, filename=filename)
        saver.save(sess, checkpoint)

        print("(Average) Loss: {}".format(np.mean(losses, axis=0)))
        print("(Average) Accuracy: {}".format(np.mean(accuracies, axis=0)))


def _eval(args):
    # Preprocess data set.
    embedding, word_vecs, labels = tt.preprocess(args.word2vec, args.labels)
    embed_shape = np.shape(embedding.syn0)
    features = embed_shape[1]
    one_hot_labels, lbl_dict = tt.one_hot(labels)
    num_labels = len(lbl_dict)
    lbl_list = tt.dict_to_list(lbl_dict)
    batch_size = len(word_vecs)

    with tf.variable_scope('lstm_model'):
        model = rnn.MultiRNNLSTM(args.rnn_size, args.depth, num_labels, batch_size, features, args.alpha)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, args.load_path)  # Load trained weights and biases.

        batch_x = word_vecs
        batch_y = one_hot_labels
        feed_dict = {model.x: batch_x, model.y: batch_y}
        _, expect, predict, eval_accuracy = sess.run([model.optimize, model.expected, model.predicted, model.accuracy], feed_dict=feed_dict)

        print("Evaluation Accuracy: {accuracy}".format(accuracy=eval_accuracy))

        # Write expected and predicted values to file in CoNLL format.
        with open(args.conll_path, mode='w+') as conll:
            for e, p in zip(expect, predict):
                line = "{e}\t{p}\n".format(e=lbl_list[e], p=lbl_list[p])
                conll.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word2Vec_RNN')
    parser.add_argument('-a', '--alpha', action='store', default=0.01, type=float)
    parser.add_argument('-b', '--board_path', action='store', default='logs', type=str)
    parser.add_argument('-c', '--conll_path', action='store')
    parser.add_argument('-d', '--depth', action='store', default=1, type=int)
    parser.add_argument('-e', '--epochs', action='store', default=5, type=int)
    parser.add_argument('-i', '--input_keep_prob', action='store', default=0.5, type=float)
    parser.add_argument('-l', '--load_path', action='store')
    parser.add_argument('-m', '--mode', action='store', required=True, choices=['learn', 'eval'])
    parser.add_argument('-r', '--rnn_size', action='store', default=32, type=int)
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
        print('Learning complete...')
    else:
        if args.load_path is None or args.conll_path is None:
            parser.error('Mode "eval" requires load path and CoNLL path.')
        print('Evaluating...')
        _eval(args)
