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

    print("{rounds} training rounds with {choices} (randomized) selections.".format(rounds=rounds, choices=np.shape(choices)[1]))

    with tf.variable_scope('lstm_model'):
        input_keep_prob = args.input_keep_prob
        if input_keep_prob == 0.0:
            input_keep_prob = None

        model = rnn.MultiRNNLSTM(args.rnn_size, args.depth, num_labels, batch_size, feature_size, args.alpha, input_keep_prob)

    saver = tf.train.Saver()
    sum_merge = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        logpath = os.path.join(args.board_path, '')
        tt.clear_logs(logpath)
        writer = tf.summary.FileWriter(logpath, graph=sess.graph)
        sess.run(init)

        losses = np.zeros((args.epochs, rounds))
        accuracies = np.zeros((args.epochs, rounds))

        # Train model.
        for epoch in np.arange(args.epochs):
            steps = np.arange(rounds)  # Each step is a batch round.

            for step in steps:
                state = np.zeros((args.depth, 2, batch_size, args.rnn_size))  # Set empty initial state for each batch.
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
        filename = os.path.splitext(os.path.basename(__file__))[0]
        checkpoint = "{path}{filename}.ckpt".format(path=path, filename=filename)
        print("Saving model variables to: {checkpoint}".format(checkpoint=checkpoint))
        saver.save(sess, checkpoint)

        print("(Average) Loss: {}".format(np.mean(losses, axis=0)))
        print("(Average) Accuracy: {}".format(np.mean(accuracies, axis=0)))


def _eval(args):
    label_dict = {'O': 0, 'Material': 1, 'Process': 2, 'Task': 3}
    label_list = ['' for n in np.arange(len(label_dict))]
    for k, v in label_dict.items():
        label_list[v] = k

    num_labels = len(label_dict)
    (sequences, labels) = tt.preprocess(args.word2vec_path, args.labels_path)  # Prepare zipped sequences of vectored words and labels.
    num_inputs = len(sequences)  # Determine number of training examples.
    feature_size = len(sequences[0][0])  # Determine word vector size.
    one_hot_labels = tt.one_hot(labels, label_dict)

    # Calculate batches.
    batch_size, steps = tt.batch_calc(num_inputs, args.target_batch_size)
    _, iter_wait = tt.batch_calc(batch_size, args.target_batch_size // 4)

    print("Evaluating {steps} batches of size {batch_size}.".format(steps=steps, batch_size=batch_size))

    with tf.variable_scope('lstm_model'):
        input_keep_prob = args.input_keep_prob
        if input_keep_prob == 0.0:
            input_keep_prob = None

        model = rnn.MultiRNNLSTM(args.rnn_size, args.depth, num_labels, batch_size, feature_size, args.alpha, input_keep_prob)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    path = os.path.join(args.load_path, '')
    filename = os.path.splitext(os.path.basename(__file__))[0]
    checkpoint = "{path}{filename}.ckpt".format(path=path, filename=filename)

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, checkpoint)  # Load trained weights and biases.

        losses = np.zeros(steps)
        accuracies = np.zeros(steps)
        conll_list = []

        # Evaluate model.
        for step in np.arange(steps):
            state = np.zeros((args.depth, 2, batch_size, args.rnn_size))  # Set empty initial state for each batch.
            chosen_seqs = np.arange(batch_size * step, batch_size * (step + 1))
            (seq_len, seq_lbl_zip) = tt.package_batch(chosen_seqs, sequences, one_hot_labels, label_dict)
            batch_x, batch_y = zip(*seq_lbl_zip)

            feed_dict = {model.x: batch_x, model.y: batch_y, model.seq_len: seq_len, model.init_state: state}

            expectation, prediction, eval_cross_entropy, eval_accuracy, state = sess.run([model.expect, model.predict, model.cross_entropy, model.accuracy, model.dynamic_state], feed_dict=feed_dict)
            losses[step] = eval_cross_entropy
            accuracies[step] = eval_accuracy

            if (step + 1) % iter_wait == 0:
                print("[Step {step:0>4d}] Loss: {loss:.5f} | Accuracy: {accuracy:.5f}".format(step=step, loss=eval_cross_entropy, accuracy=eval_accuracy))

            zip_list = []
            for e, p in zip(expectation, prediction):
                zip_list.append((label_list[e], label_list[p]))
            conll_list.extend(zip_list)

        print("(Average) Loss: {}".format(np.mean(losses, axis=0)))
        print("(Average) Accuracy: {}".format(np.mean(accuracies, axis=0)))

        path = os.path.join(args.conll_path, '')
        filename = os.path.splitext(os.path.basename(__file__))[0]
        conll_file = "{path}{filename}.log".format(path=path, filename=filename)
        # Write expected and predicted values to file in CoNLL format.
        with open(conll_file, mode='w+') as conll:
            for pairing in conll_list:
                line = "{e}\t{p}\n".format(e=pairing[0], p=pairing[1])
                conll.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word2Vec_RNN')
    parser.add_argument('-a', '--alpha', action='store', default=0.0001, type=float)
    parser.add_argument('-b', '--board_path', action='store', default='logs', type=str)
    parser.add_argument('-c', '--conll_path', action='store')
    parser.add_argument('-d', '--depth', action='store', default=1, type=int)
    parser.add_argument('-e', '--epochs', action='store', default=5, type=int)
    parser.add_argument('-i', '--input_keep_prob', action='store', default=0.20, type=float)
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
