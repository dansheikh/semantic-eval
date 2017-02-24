#!/usr/bin/env python
import argparse
import numpy as np
import tensorflow as tf
import tensor_tools as tt
import rnn


def _learn(args):
    # Preprocess data set.
    embedding, word_vecs, lbls = tt.preprocess(args.word2vec, args.labels)
    embed_shape = np.shape(embedding.syn0)
    features = embed_shape[1]
    one_hot_lbls, lbl_dict = tt.one_hot(lbls)
    num_labels = len(lbl_dict)
    batch_size, num_steps = tt.batch_calc(len(word_vecs), args.origin)
    _, iter_wait = tt.batch_calc(num_steps, args.origin)

    with tf.variable_scope('lstm_model'):
        lstm = rnn.MultiRNNLSTM(args.rnn_size, args.depth, num_labels, batch_size, features, args.alpha)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Train model.
        for epoch in np.arange(args.epochs):
            steps = np.arange(num_steps)
            np.random.shuffle(steps)
            for idx, time_step in enumerate(steps):
                batch_x = word_vecs[time_step * batch_size:(time_step + 1) * batch_size]
                batch_y = one_hot_lbls[time_step * batch_size:(time_step + 1) * batch_size]
                feed_dict = {lstm.x: batch_x, lstm.y: batch_y}
                _, expect, predict, train_accuracy = sess.run([lstm.optimize, lstm.expected, lstm.predicted, lstm.accuracy], feed_dict=feed_dict)

                if (idx + 1) % iter_wait == 0:
                    print("[Epoch {epoch}, Iteration {idx}] Accuracy: {accuracy}".format(epoch=epoch, idx=idx, accuracy=train_accuracy))
                    print("[Expected]\t{expect}".format(expect=expect))
                    print("[Predicted]\t{predict}".format(predict=predict))

        print("Trained over {} batches and steps {}".format(batch_size, num_steps))
        checkpoint = "{path}.ckpt".format(path=args.save_path)
        saver.save(sess, checkpoint)


def _eval(args):
    # Preprocess data set.
    embedding, word_vecs, lbls = tt.preprocess(args.word2vec, args.labels)
    embed_shape = np.shape(embedding.syn0)
    features = embed_shape[1]
    one_hot_lbls, lbl_dict = tt.one_hot(lbls)
    num_labels = len(lbl_dict)
    lbl_list = tt.dict_to_list(lbl_dict)
    batch_size = len(word_vecs)

    with tf.variable_scope('lstm_model'):
        lstm = rnn.MultiRNNLSTM(args.rnn_size, args.depth, num_labels, batch_size, features, args.alpha)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, args.load_path)  # Load trained weights and biases.

        batch_x = word_vecs
        batch_y = one_hot_lbls
        feed_dict = {lstm.x: batch_x, lstm.y: batch_y}
        _, expect, predict, eval_accuracy = sess.run([lstm.optimize, lstm.expected, lstm.predicted, lstm.accuracy], feed_dict=feed_dict)

        print("Evaluation Accuracy: {accuracy}".format(accuracy=eval_accuracy))

        # Write expected and predicted values to file in CoNLL format.
        with open(args.conll_path, mode='w+') as conll:
            for e, p in zip(expect, predict):
                line = "{e}\t{p}\n".format(e=lbl_list[e], p=lbl_list[p])
                conll.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word2Vec_RNN')
    parser.add_argument('-a', '--alpha', action='store', default=0.01, type=float)
    parser.add_argument('-c', '--conll_path', action='store')
    parser.add_argument('-d', '--depth', action='store', default=3, type=int)
    parser.add_argument('-e', '--epochs', action='store', default=5, type=float)
    parser.add_argument('-k', '--keep_prob', action='store', default=0.5, type=float)
    parser.add_argument('-l', '--load_path', action='store')
    parser.add_argument('-m', '--mode', action='store', required=True, choices=['learn', 'eval'])
    parser.add_argument('-o', '--origin', action='store', default=25, type=int)
    parser.add_argument('-r', '--rnn_size', action='store', default=32, type=int)
    parser.add_argument('-s', '--save_path', action='store')
    parser.add_argument('word2vec')
    parser.add_argument('labels')
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
