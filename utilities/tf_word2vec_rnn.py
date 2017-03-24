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
    batch_size = args.target_batch_size

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

        losses = []
        accuracies = []
        scores = []
        epoch_cnt = 0
        step_cnt = 0
        plateau_cnt = 0
        learning = True

        # Train model.
        while learning:
            # Add placeholder lists.
            losses.append([])
            accuracies.append([])
            scores.append([])

            rounds, choices = tt.random_choices(num_inputs, batch_size)  # Calculate batches.
            steps = np.arange(rounds)  # Each step is a batch of sequences.

            for step in steps:
                step_cnt += 1  # Increase step counter.
                state = np.zeros((args.depth, 2, batch_size, args.rnn_size))  # Set empty initial state for each batch.
                chosen_seqs = choices[step]
                (seq_len, seq_lbl_zip) = tt.package_batch(chosen_seqs, sequences, one_hot_labels, label_dict)
                batch_x, batch_y = zip(*seq_lbl_zip)

                feed_dict = {model.x: batch_x, model.y: batch_y, model.seq_len: seq_len, model.init_state: state}

                _, logits, train_cross_entropy, train_accuracy, state, summary = sess.run([model.optimize, model.logits, model.cross_entropy, model.accuracy, model.dynamic_state, sum_merge], feed_dict=feed_dict)

                # Check if learning has plateaued.
                if plateau_cnt == 3:
                    print('Learning plateaued.')
                    learning = False
                    break
                elif step_cnt * batch_size >= args.kill_zone:
                    print('Learning timeout.')
                    learning = False
                    break
                else:
                    if len(losses[epoch_cnt]) > 0 and abs(losses[epoch_cnt][-1] - train_cross_entropy) < args.alpha:
                        plateau_cnt += 1
                    else:
                        plateau_cnt = 0

                    # Update performance logs.
                    losses[epoch_cnt].append(train_cross_entropy)
                    accuracies[epoch_cnt].append(train_accuracy)
                    scores[epoch_cnt].append(np.array(logits))

                if learning and (step_cnt * batch_size) % batch_size == 0:
                    print("[Step {steps:0>3d}] Loss: {loss:.5f} | Accuracy: {accuracy:.5f}".format(steps=(step_cnt * batch_size), loss=train_cross_entropy, accuracy=train_accuracy))

                writer.add_summary(summary, step)

            if learning:
                epoch_cnt += 1  # Increment epoch counter.

        scores = np.array(scores)  # Convert scores list to array.
        path = os.path.join(args.save_path, '')
        filename = os.path.splitext(os.path.basename(__file__))[0]
        checkpoint = "{path}{filename}.ckpt".format(path=path, filename=filename)
        print("Saving model variables to: {checkpoint}".format(checkpoint=checkpoint))
        saver.save(sess, checkpoint)

        zip_loss_accuracy = zip(tt.epoch_mean(losses), tt.epoch_mean(accuracies))

        for zip_loss, zip_accuracy in zip_loss_accuracy:
            if zip_loss is not None and zip_accuracy is not None:
                print("(Average) Loss: {loss:.5f} | Accuracy {accuracy:.5f}".format(loss=zip_loss, accuracy=zip_accuracy))

        if args.peek:
            logits_path = os.path.join(args.peek, '')
            logits_filename = os.path.splitext(os.path.basename(__file__))[0]
            logits_file = "{path}{filename}.npy".format(path=logits_path, filename=logits_filename)
            if os.path.exists(logits_file):
                os.remove(logits_file)

            print("Saving scores to {logits_file}".format(logits_file=logits_file))
            np.save(logits_file, np.array(scores))


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
    parser.add_argument('-b', '--board_path', action='store', default='logs/', type=str)
    parser.add_argument('-c', '--conll_path', action='store')
    parser.add_argument('-d', '--depth', action='store', default=1, type=int)
    parser.add_argument('-i', '--input_keep_prob', action='store', default=0.20, type=float)
    parser.add_argument('-k', '--kill_zone', action='store', default=1000000, type=int)
    parser.add_argument('-l', '--load_path', action='store')
    parser.add_argument('-m', '--mode', action='store', required=True, choices=['learn', 'eval'])
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
