#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import gensim


def _main():
    """Converts target words from Conll formatted files to vectorized model representation."""
    parser = argparse.ArgumentParser(description='Converter')
    parser.add_argument('-s', '--sep', action='store', default='\t')
    parser.add_argument('-m', '--mode', action='store', required=True, choices=['cbow', 'skipgram'])
    parser.add_argument('-c', '--count', action='store', default=1, type=int)
    parser.add_argument('-l', '--length', action='store', default=100, type=int)
    parser.add_argument('-w', '--workers', action='store', default=1, type=int)
    parser.add_argument('-j', '--join', action='store', default=False, type=bool)
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    args = parser.parse_args()

    sg = 0
    if args.mode == 'skipgram':
        sg = 1

    sentences = list()

    try:
        dir = None
        if os.path.isdir(args.data_path):
            dir = os.listdir(args.data_path)

        for file in dir:
            print('Reading ' + file)
            tmp = list()
            with open(os.path.join(args.data_path, file), mode='r') as input:
                content = enumerate(input)

                for idx, line in content:
                    sentence = list()

                    for id, word in enumerate(line.rstrip(os.linesep).split(args.sep)):
                        if id == 1:
                            sentence.append(word)

                    sentences.append(sentence)

        msg = "Building {mode} vocabulary...".format(mode=args.mode)
        print(msg)
        model = gensim.models.Word2Vec(sentences, sg=sg, workers=args.workers, min_count=args.count, size=args.length)
        model.save(args.save_path)

    except IOError as err:
        stmt = "File Operation Failed: {error}".format(error=err.strerror)
        print(stmt)


if __name__ == '__main__':
    _main()
