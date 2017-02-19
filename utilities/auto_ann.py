#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import xml.sax
import json
import re
import xml_utils
import collections
import nltk.corpus


def annotate(handler, outpath, labels):
    regex = r"[\w]+|\[|\]|\,|\."

    with open(outpath, 'a+') as file:
        for idx, text in handler.text.items():
            words = re.findall(regex, text)
            previous_pair = (None, None)

            for word in words:
                if word.lower() in labels:
                    # Check if word is stopword.
                    if labels[word.lower()]['stopword']:
                        # Check if preceding word in following list and tagged.
                        if  previous_pair[0] in labels[word.lower()]['following'] and previous_pair[1] != 'O':
                            feature = "{word}\t{label}\n".format(word=word, label=labels[word.lower()]['tag'])
                            previous_pair = (word, labels[word.lower()]['tag'])
                        else:
                            feature = "{word}\tO\n".format(word=word)
                            previous_pair = (word, 'O')
                    else:
                        feature = "{word}\t{label}\n".format(word=word, label=labels[word.lower()]['tag'])
                        previous_pair = (word, labels[word.lower()]['tag'])

                    file.write(feature)
                else:
                    feature = "{word}\tO\n".format(word=word)
                    previous_pair = (word, 'O')
                    file.write(feature)


def parse_xml(inpath, outpath, labels):
    # Create an XMLReader
    parser = xml.sax.make_parser()
    # Turn off namespaces.
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    handler = xml_utils.PubHandler()
    parser.setContentHandler(handler)

    # Parse document.
    parser.parse(inpath)
    # Annotate text.
    annotate(handler, outpath, labels)


def parse_xmls(inpath, outpath, labels):
    dir = os.listdir(inpath)

    for file in dir:
        if not file.endswith(".xml"):
            continue

        parse_xml(os.path.join(inpath, file), outpath, labels)


def extract_labels(path, sep=' '):
    labels = dict()
    stops = set(nltk.corpus.stopwords.words('english'))

    with open(path, mode='r') as file:
        previous_pair = tuple([None, None])

        for idx, line in enumerate(file):
            content = line.strip()

            if content:
                elements = content.split(sep)

                if len(elements) == 2 and elements[1] != 'O':
                    if elements[0].lower() not in labels:
                        labels[elements[0].lower()] = dict()
                        labels[elements[0].lower()]['tag'] = elements[1]
                        labels[elements[0].lower()]['following'] = list()

                        if elements[0].lower() in stops:
                            labels[elements[0].lower()]['stopword'] = True
                            labels[elements[0].lower()]['following'].append(previous_pair[0])
                        else:
                            labels[elements[0].lower()]['stopword'] = False

                    else:
                        if elements[0].lower() in stops:
                            labels[elements[0].lower()]['following'].append(previous_pair[0])

            previous_pair = (elements[0], elements[1])

    return labels


def _launch():
    parser = argparse.ArgumentParser(description='Auto-annotate')
    parser.add_argument('-s', '--sep', action='store', default=' ')
    parser.add_argument('sample')
    parser.add_argument('inpath')
    parser.add_argument('outpath')
    args = parser.parse_args()

    # Download stopwords.
    nltk.download('stopwords')

    # Extract labels from tagged-by-hand file.
    labels = extract_labels(args.sample, args.sep)
    print(json.dumps(labels, ensure_ascii=False))

    if os.path.isdir(args.inpath):
        parse_xmls(args.inpath, args.outpath, labels)
    else:
        parse_xml(args.inpath, args.outpath, labels)


if __name__ == '__main__':
    _launch()
