#!/usr/bin/env python

import argparse
import xml_utils


def _launch():
    parser = argparse.ArgumentParser(description="Parse XML")
    parser.add_argument('-a', '--all', action='store_true', default=False)
    parser.add_argument('path')
    args = parser.parse_args()

    if args.all:
        xml_utils.parseXMLAll(args.path)
    else:
        xml_utils.parseXML(args.path)

if __name__ == '__main__':
    _launch()
