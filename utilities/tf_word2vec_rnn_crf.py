#!/usr/bin/env python

import argparse
import os

import numpy as np
import tensorflow as tf

import rnn_crf
import tensor_tools as tt


def _learn(args):
    label_dict = None
    if args.nontyped:
        label_dict = {'O': 0, 'Keyphrase': 1}
    else:
        label_dict = {'O': 0, 'Material': 1, 'Process': 2, 'Task': 3}
    num_labels = len(label_dict)
    (sequences, labels) = tt.preprocess(args.word2vec_path, args.labels_path)  # Prepare zipped sequences of vectored words and labels.
    scalar_labels = tt.numeric_labels(labels, label_dict)
    num_inputs = len(sequences)  # Determine number of training examples.
    num_feats = len(sequences[0][0])  # Determine word vector size.
    seq_lens = [len(sequence) for sequence in sequences]
    seq_lens = np.array(seq_lens)

    # Pad sentences and labels.
    pad_seqs, pad_lbls = tt.pad_data(sequences, scalar_labels, label_dict)
    num_words = np.shape(pad_seqs)[1]
    num_batches = num_inputs // args.target_batch_size

    with tf.variable_scope('lstm_model'):
        input_keep_prob = args.input_keep_prob
        if input_keep_prob == 0.0:
            input_keep_prob = None

        if args.bi:
            model = rnn_crf.RCRF(args.rnn_size, args.depth, num_labels, num_words, num_feats, args.optimizer, args.alpha, bi=True, input_keep_prob=input_keep_prob)
        else:
            model = rnn_crf.RCRF(args.rnn_size, args.depth, num_labels, num_words, num_feats, args.optimizer, args.alpha, bi=False, input_keep_prob=input_keep_prob)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        losses = []
        accuracies = []
        scores = []
        epoch = 0
        plateau_cnt = 0
        learning = True

        # Train model.
        while learning:
            # Add placeholder lists.
            losses.append([])
            accuracies.append([])
            scores.append([])

            for batch_idx in np.arange(num_batches):
                total_lbls = 0
                correct_lbls = 0
                train_accuracy = None
                batch_x = None
                batch_y = None
                batch_seq_lens = None

                if batch_idx < num_batches:
                    batch_x = pad_seqs[(batch_idx * args.target_batch_size): ((batch_idx + 1) * args.target_batch_size)]
                    batch_y = pad_lbls[(batch_idx * args.target_batch_size): ((batch_idx + 1) * args.target_batch_size)]
                    batch_seq_lens = seq_lens[(batch_idx * args.target_batch_size): ((batch_idx + 1) * args.target_batch_size)]
                else:
                    batch_x = pad_seqs[(batch_idx * args.target_batch_size):]
                    batch_y = pad_lbls[(batch_idx * args.target_batch_size):]
                    batch_seq_lens = seq_lens[(batch_idx * args.target_batch_size):]

                current_batch_size = np.shape(batch_seq_lens)[0]
                state = np.zeros((args.depth, 2, current_batch_size, args.rnn_size))  # Set empty initial state for each batch.

                feed_dict = {model.x: batch_x, model.y: batch_y, model.seq_lens: batch_seq_lens, model.init_state: state}

                train_logits, train_trans_params, train_loss, state, _, = sess.run([model.logits, model.trans_params, model.loss, model.dynamic_state, model.optimize], feed_dict=feed_dict)

                for (logit, lbl, seq_len) in zip(train_logits, batch_y, batch_seq_lens):
                    logit = logit[:seq_len]
                    lbl = lbl[:seq_len]
                    viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, train_trans_params)

                    correct_lbls += np.sum(np.equal(viterbi_seq, lbl))
                    total_lbls += seq_len

                train_accuracy = 100 * correct_lbls / float(total_lbls)

                accuracies[-1].append(train_accuracy)
                losses[-1].append(train_loss)

            # Check loss diff
            if len(losses) > 1:
                loss_diff = np.mean(losses[-2], axis=0) - np.mean(losses[-1], axis=0)
                if abs(loss_diff) < args.epsilon:
                    plateau_cnt += 1
                else:
                    plateau_cnt = 0

            # Check if learning has plateaued.
            if plateau_cnt >= 3:
                print('Learning plateaued.')
                learning = False
                break
            elif epoch >= args.kill_zone:
                print('Learning timeout.')
                learning = False
                break

            if learning and (epoch + 1) % 100:
                print("[Epoch {epoch}] Loss: {loss:.5f} | Accuracy: {accuracy:.5f}".format(epoch=epoch, loss=np.mean(losses[-1], axis=0), accuracy=np.mean(accuracies[-1], axis=0)))
            epoch += 1  # Increase epoch counter.

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

    num_labels = len(label_dict)
    (sequences, labels) = tt.preprocess(args.word2vec_path, args.labels_path)  # Prepare zipped sequences of vectored words and labels.
    scalar_labels = tt.numeric_labels(labels, label_dict)
    num_inputs = len(sequences)  # Determine number of training examples.
    num_feats = len(sequences[0][0])  # Determine word vector size.
    seq_lens = [len(sequence) for sequence in sequences]
    seq_lens = np.array(seq_lens)

    # Pad sentences and labels.
    pad_seqs, pad_lbls = tt.pad_data(sequences, scalar_labels, label_dict)
    num_words = np.shape(pad_seqs)[1]
    num_batches = num_inputs // args.target_batch_size

    with tf.variable_scope('lstm_model'):
        input_keep_prob = args.input_keep_prob
        if input_keep_prob == 0.0:
            input_keep_prob = None

        if args.bi:
            model = rnn_crf.RCRF(args.rnn_size, args.depth, num_labels, num_words, num_feats, args.optimizer, args.alpha, bi=True, input_keep_prob=input_keep_prob)
        else:
            model = rnn_crf.RCRF(args.rnn_size, args.depth, num_labels, num_words, num_feats, args.optimizer, args.alpha, bi=False, input_keep_prob=input_keep_prob)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    path = os.path.join(args.load_path, '')
    filename = os.path.splitext(os.path.basename(__file__))[0]
    checkpoint = "{path}{filename}.ckpt".format(path=path, filename=filename)

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, checkpoint)  # Load trained weights and biases.

        for batch_idx in np.arange(num_batches):
            if batch_idx < num_batches:
                batch_x = pad_seqs[(batch_idx * args.target_batch_size): ((batch_idx + 1) * args.target_batch_size)]
                batch_y = pad_lbls[(batch_idx * args.target_batch_size): ((batch_idx + 1) * args.target_batch_size)]
                batch_seq_lens = seq_lens[(batch_idx * args.target_batch_size): ((batch_idx + 1) * args.target_batch_size)]
            else:
                batch_x = pad_seqs[(batch_idx * args.target_batch_size):]
                batch_y = pad_lbls[(batch_idx * args.target_batch_size):]
                batch_seq_lens = seq_lens[(batch_idx * args.target_batch_size):]

            current_batch_size = np.shape(batch_seq_lens)[0]

            state = np.zeros((args.depth, 2, current_batch_size, args.rnn_size))  # Set empty initial state for each batch.
            feed_dict = {model.x: batch_x, model.y: batch_y, model.seq_lens: batch_seq_lens, model.init_state: state}

            eval_logits, eval_trans_params, state = sess.run([model.logits, model.trans_params, model.dynamic_state], feed_dict=feed_dict)

            for (logit, lbl, seq_len) in zip(eval_logits, batch_y, batch_seq_lens):
                logit = logit[:seq_len]
                lbl = lbl[:seq_len]
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, eval_trans_params)

                path = os.path.join(args.conll_path, '')
                filename = os.path.splitext(os.path.basename(__file__))[0]
                conll_file = "{path}{filename}.log".format(path=path, filename=filename)
                # Write expected and predicted values to file in CoNLL format.
                with open(conll_file, mode='a+') as conll:
                    for expect, predict in zip(lbl, viterbi_seq):
                        line = "{e}\t{p}\n".format(e=label_list[expect], p=label_list[predict])
                        conll.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word2Vec_RNN')
    parser.add_argument('-a', '--alpha', action='store', default=0.001, type=float)
    parser.add_argument('-b', '--board_path', action='store', default='logs/', type=str)
    parser.add_argument('--bi', action='store_true')
    parser.add_argument('-c', '--conll_path', action='store')
    parser.add_argument('-d', '--depth', action='store', default=1, type=int)
    parser.add_argument('-e', '--epsilon', action='store', default=0.001, type=float)
    parser.add_argument('-i', '--input_keep_prob', action='store', default=0.20, type=float)
    parser.add_argument('-k', '--kill_zone', action='store', default=1000000, type=int)
    parser.add_argument('-l', '--load_path', action='store')
    parser.add_argument('-m', '--mode', action='store', required=True, choices=['learn', 'eval'])
    parser.add_argument('--nontyped', action='store_true')
    parser.add_argument('-o', '--optimizer', action='store', choices=['adam', 'rms', 'sgd'], default='adam')
    parser.add_argument('-p', '--peek', action='store', default='', type=str)
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
        print('Learning complete.')
    else:
        if args.load_path is None or args.conll_path is None:
            parser.error('Mode "eval" requires load path and CoNLL path.')

        print('Evaluating...')
        _eval(args)
        print('Evaluation complete.')
