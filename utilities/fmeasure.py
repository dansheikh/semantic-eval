#!/usr/bin/env python

import sys

with open(sys.argv[1]) as file:
    lines = [line.strip().split() for line in file]

precision, recall, selects, actuals = (0, 0, 0, 0)

for pair in lines:
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F-Measure = 2PR / (P + R)

    if len(pair) == 2:
        if pair[1].lower() != 'o':
            if pair[0].lower() == pair[1].lower():
                precision += 1
            selects += 1

        if pair[0].lower() != 'o':
            if pair[0].lower() == pair[1].lower():
                recall += 1
            actuals += 1

precision = precision/selects
recall = recall/actuals
fmeasure = (2 * precision * recall) / (precision + recall)

print("F-Score: {:.5f} [Precision: {:.5f} | Recall: {:.5f}]".format(fmeasure, precision, recall))
