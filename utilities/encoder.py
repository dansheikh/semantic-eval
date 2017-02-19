#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import embed
import gensim


def _launch():
    regex = re.compile('([S|s]kip[-|_|\s]{0,1}gram[s]{0,1})')
    parser = argparse.ArgumentParser(description='Encode')
    parser.add_argument('-m', '--mode', action='store', default='cbow')
    parser.add_argument('-w', '--workers', action='store', default=1, type=int)
    parser.add_argument('xmlpath')
    parser.add_argument('savepath')
    args = parser.parse_args()

    sg = 0  # Set default training algorithm to CBoW.
    workers = args.workers

    if regex.search(args.mode) is not None:
        sg = 1

    passages = embed.Passages(args.xmlpath)

    model = gensim.models.Word2Vec(passages, sg=sg, workers=workers)
    model.save(args.savepath)


if __name__ == '__main__':
    _launch()
