#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pprint
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def _rand_color():
    def randi():
        return np.random.randint(0, 255)

    return "#{:02X}{:02X}{:02X}".format(randi(), randi(), randi())


def _main(sent_file, infile):
    pp = pprint.PrettyPrinter()
    tags = ['keyphrase', 'material', 'process', 'task']
    sent_lens = np.load(sent_file)
    sent_lens_cnt = len(sent_lens)
    sent_stats = {(n // 10): {0: 0, 1: 0} for n in sent_lens}
    phrase_stats = {}
    line_cnt = 0

    try:
        with open(infile, mode='r') as input:
            phrase = False
            expected = []
            predicted = []
            sent_marker = 0
            sent_cnt = sent_lens[sent_marker]

            for idx, line in enumerate(input):
                if line_cnt > sent_cnt - 1:
                    if sent_marker < sent_lens_cnt - 1:
                        sent_marker += 1
                        sent_cnt += sent_lens[sent_marker]

                content = line.rstrip(os.linesep).split('\t')

                if not content[0].strip():
                    continue
                else:
                    line_cnt += 1

                if content[0].lower() in tags:
                    if phrase:
                        expected.append(content[0])
                        predicted.append(content[1])
                    if not phrase:
                        phrase = True
                        expected.append(content[0])
                        predicted.append(content[1])
                elif len(expected) > 0 or len(predicted) > 0:
                    sent_bin = sent_lens[sent_marker] // 10
                    phrase_bin = len(expected) // 2

                    if phrase_bin in phrase_stats:
                        expected = np.array(expected)
                        predicted = np.array(predicted)
                        is_correct = (expected == predicted).all()
                        key = int(is_correct)
                        sent_stats[sent_bin][key] += 1
                        phrase_stats[phrase_bin][key] += 1
                    else:
                        phrase_stats[phrase_bin] = {0: 0, 1: 0}
                        expected = np.array(expected)
                        predicted = np.array(predicted)
                        is_correct = (expected == predicted).all()
                        key = int(is_correct)
                        sent_stats[sent_bin][key] += 1
                        phrase_stats[phrase_bin][key] += 1

                    # Reset.
                    expected = []
                    predicted = []
                    phrase = False

            pp.pprint(sent_stats)
            pp.pprint(phrase_stats)

            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.grid()
            ax2.grid()

            ax1.set_title('Sentence Level Performance')
            ax2.set_title('Phrase Level Performance')

            ax1.set_xlabel('Sentence Length')
            ax1.set_ylabel('Tally')
            ax2.set_xlabel('Phrase Length')
            ax2.set_ylabel('Tally')

            sent_bars = sorted([key for key in sent_stats.keys()])
            phrase_bars = sorted([key for key in phrase_stats.keys()])

            sent_colors = [_rand_color() for _ in np.arange(2)]
            phrase_colors = [_rand_color() for _ in np.arange(2)]

            sent_lists = [[], []]
            phrase_lists = [[], []]

            sent_xticks = [str((tick + 1) * 10) for tick in sent_bars]
            phrase_xticks = [str((tick + 1) * 2) for tick in phrase_bars]

            ax1.set_xticks(np.arange(len(sent_bars)))
            ax2.set_xticks(np.arange(len(phrase_bars)))

            ax1.set_xticklabels(sent_xticks)
            ax2.set_xticklabels(phrase_xticks)

            for label in ax1.get_xticklabels():
                label.set_fontsize(6)

            for label in ax2.get_xticklabels():
                label.set_fontsize(6)

            ax1.grid(True)
            ax2.grid(True)

            for s in sent_bars:
                sent_lists[0].append(sent_stats[s][0])
                sent_lists[1].append(sent_stats[s][1])

            for p in phrase_bars:
                phrase_lists[0].append(phrase_stats[p][0])
                phrase_lists[1].append(phrase_stats[p][1])

            ax1.bar(sent_bars, sent_lists[1], align='center', color=sent_colors[0], label='Correct')
            ax1.bar(sent_bars, sent_lists[0], bottom=sent_lists[1], align='center', color=sent_colors[1], label='Incorrect')
            ax2.bar(phrase_bars, phrase_lists[1], align='center', color=phrase_colors[0], label='Correct')
            ax2.bar(phrase_bars, phrase_lists[0], bottom=phrase_lists[1], align='center', color=phrase_colors[1], label='Incorrect')

            ax1.legend()
            ax2.legend()

            dirpath = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
            filename = os.path.splitext(os.path.basename(__file__))[0]
            img_name = "{}.png".format(filename)
            img_loc = os.path.join(dirpath, 'images', img_name)
            print('Saving plots...')
            fig.tight_layout()
            fig.savefig(img_loc)
            plt.close(fig)
            print('Plots saved.')

    except IOError as err:
        stmt = "File Operation Failed: {error}".format(error=err.strerror)
        print(stmt)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise AttributeError('Input and Output file paths required.')

    _main(sys.argv[1], sys.argv[2])
