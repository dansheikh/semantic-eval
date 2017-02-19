#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import os
import gensim
import numpy as np


def _main():
    """Uses word-to-vector model to produce Conll formatted files with vectorized target words."""

    parser = argparse.ArgumentParser(description='Auto-annotate')
    parser.add_argument('-s', '--sep', action='store', default='\t')
    parser.add_argument('embed_loc')
    parser.add_argument('read_file')
    parser.add_argument('write_file')
    args = parser.parse_args()

    model = gensim.models.Word2Vec.load(args.embed_loc)

    try:
        with open(args.read_file, mode='r') as input, open(args.write_file, mode='w') as output:
            content = enumerate(input)

            for idx, line in content:
                for id, word in enumerate(line.rstrip(os.linesep).split(args.sep)):
                    if id == 0:
                        entry = "{word}{sep}".format(word=word, sep=args.sep)
                        output.write(entry)
                    if id == 1:
                        entry = "{word}".format(word=np.array2string(model[word]))
                        output.write(entry)
                    if id == 2:
                        entry = "{sep}{word}".format(sep=args.sep, word=word)
                        output.write(entry)
                output.write(os.linesep)

    except IOError as err:
        stmt = "File Operation Failed: {error}".format(error=err.strerror)
        print(stmt)


if __name__ == '__main__':
    _main()
