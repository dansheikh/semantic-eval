#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pprint
import sys
import numpy as np


def _main(infile, outfile):
    START = '__BOS__'
    STOP = '__EOS__'
    seqs = []
    cnt = 0
    pp = pprint.PrettyPrinter()

    try:
        with open(infile, mode='r') as input:
            for idx, line in enumerate(input):
                content = line.rstrip(os.linesep).split('\t')

                if len(content) == 3 and content[-1] == START:
                    cnt += 1
                elif len(content) == 3 and content[-1] == STOP:
                    cnt += 1
                    seqs.append(cnt)
                    cnt = 0
                else:
                    cnt += 1

        pp.pprint(seqs)
        filename = "{}.npy".format(outfile)
        np.save(filename, np.array(seqs))

    except IOError as err:
        stmt = "File Operation Failed: {error}".format(error=err.strerror)
        print(stmt)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise AttributeError('Input and Output file paths required.')

    _main(sys.argv[1], sys.argv[2])
