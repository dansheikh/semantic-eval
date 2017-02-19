#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os


def _launch(in_file, out_file):
    try:
        with open(in_file, mode='r') as input, open(out_file, mode='w') as output:
            alpha_seq = '__BOS__'
            omega_seq = '__EOS__'
            tmp = None
            content = enumerate(input)

            for idx, line in content:
                if idx == 0:
                    words = line.rstrip(os.linesep).split('\t')
                    reversed_words = words[::-1]
                    entry = "{line}\t{tag}".format(line='\t'.join(reversed_words), tag=alpha_seq)
                    output.write(entry)
                    tmp = alpha_seq
                elif len(line) == 1 and line == os.linesep:
                    words = line.rstrip(os.linesep).split('\t')
                    reversed_words = words[::-1]
                    entry = "\t{tag}{sep}{line}".format(tag=omega_seq, sep=os.linesep, line='\t'.join(reversed_words))
                    output.write(entry)
                    tmp = line
                elif tmp == os.linesep:
                    words = line.rstrip(os.linesep).split('\t')
                    reversed_words = words[::-1]
                    entry = "{sep}{line}\t{tag}".format(sep=os.linesep, line='\t'.join(reversed_words), tag=alpha_seq)
                    output.write(entry)
                    tmp = alpha_seq
                else:
                    words = line.rstrip(os.linesep).split('\t')
                    reversed_words = words[::-1]
                    entry = "{sep}{line}".format(sep=os.linesep, line='\t'.join(reversed_words))
                    output.write(entry)
                    tmp = line

    except IOError as err:
        stmt = "File Operation Failed: {error}".format(error=err.strerror)
        print(stmt)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise AttributeError('Input and Output files required.')
    else:
        _launch(sys.argv[1], sys.argv[2])
