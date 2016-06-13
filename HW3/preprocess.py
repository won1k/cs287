#!/usr/bin/env python

"""
Language modeling preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import csv

# Global preprocessing variables
unk = 2
start = 3
end = 4

# Your preprocessing, features construction, and word2vec code.

def get_vocab(word_dict):
    """
    Open word_dict and convert to dict.
    """
    word_to_idx = {}
    with open(word_dict, 'r') as f:
            f = csv.reader(f, delimiter = '\t')
            for row in f:
                word_to_idx[row[1]] = int(row[0])
    return word_to_idx

def convert_data(data_name, word_to_idx, n):
    """
    Convert raw text to indices.
    Add n-gram windows and transform to dataset.
    """
    input_windows = []
    output = []
    with open(data_name, 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            sentLen = len(words)
            sent = [start] * (n-1) + [word_to_idx[word] for word in words] + [end] # start/end markers and padding
            for i in range(sentLen + 1): # need to predict end marker
                output.append(sent[i + (n-1)])
                input_windows.append(sent[i:(i + (n-1))])
    return np.array(input_windows, dtype = np.int32), np.array(output, dtype = np.int32)

def convert_test_data(data_name, word_to_idx, n):
    """
    Convert test data to indices.
    """
    contexts = []
    candidates = []
    with open(data_name, 'r') as f:
        for line in f:
            words = line.strip().split(' ') # remove 'C'/'Q' labels and last blank
            if words[0] == 'C':
                words = words[1:-1] # remove 'C'/'Q' labels and last blank
                contLen = len(words)
                if contLen < n - 1:
                    context = [start] * ((n-1) - contLen) + [word_to_idx[word] for word in words]
                else:
                    context = [word_to_idx[word] for word in words][-(n-1):]
                contexts.append(context)
            else:
                words = words[1:]
                candidates.append([word_to_idx[word] for word in words])
    return np.array(contexts, dtype = np.int32), np.array(candidates, dtype = np.int32)


FILE_PATHS = {"PTB": ("data/train.txt",
                      "data/valid.txt",
                      "data/test_blanks.txt",
                      "data/words.dict"),
            "PTB1000": ("data/train.1000.txt",
                      "data/dev.1000.txt",
                      "data/test_blanks.txt",
                      "data/words.1000.dict")}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set (PTB or PTB1000)",
                        type=str)
    parser.add_argument('n', help="n-gram length", type=int)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test, word_dict = FILE_PATHS[dataset]
    n = args.n

    # Get word dict
    word_to_idx = get_vocab(word_dict)
    nclasses = len(word_to_idx.keys())

    # Convert data with n-gram windows
    train_input, train_output = convert_data(train, word_to_idx, n)
    if valid:
        valid_input, valid_output = convert_data(valid, word_to_idx, n)
    if test:
        test_input, test_candidates = convert_test_data(test, word_to_idx, n)

    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
            f['test_candidates'] = test_candidates
        f['nclasses'] = np.array([nclasses], dtype=np.int32)
        f['n'] = np.array([n], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
