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

def convert_data(data_name, word_to_idx, batch_size):
    """
    Convert raw text to indices.
    Transform to (total / batch_size) x batch_size matrix.
    """
    input_idx = []
    output_idx = []
    with open(data_name, 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            sent = [start] + [word_to_idx[word] for word in words] + [end] # start/end markers and padding
            input_idx += sent[:-1] # or input_idx += sent
            output_idx += sent[1:] # or output_idx += (sent[1:] + sent[0])
    input_rnn = []
    output_rnn = []
    input_row = []
    output_row = []
    for i, idx in enumerate(input_idx):
        if len(input_row) < batch_size:
            input_row.append(idx)
            output_row.append(output_idx[i])
        else:
            input_rnn.append(input_row)
            output_rnn.append(output_row)
            input_row = [idx]
            output_row = [output_idx[i]]
    return np.array(input_rnn, dtype = np.int32), np.array(output_rnn, dtype = np.int32)

def convert_test_data(data_name, word_to_idx): # fix this for batch_size
    """
    Convert test data to indices.
    """
    contexts = []
    candidates = []
    # first get max context len
    max_context = 0
    with open(data_name, 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            if words[0] == 'C':
                max_context = max(max_context, len(words[1:-1]))
    with open(data_name, 'r') as f:
        for line in f:
            words = line.strip().split(' ') # remove last blank
            idxs = []
            for word in words:
                try:
                    idxs.append(word_to_idx[word])
                except:
                    idxs.append(unk)
            if words[0] == 'C':
                idxs = idxs[1:-1] # remove 'C'/'Q' labels and last blank
                cont_len = len(idxs)
                if cont_len < max_context:
                    context = [unk] * (max_context - cont_len) + idxs
                else:
                    context = idxs
                contexts.append(context)
            else:
                idxs = idxs[1:]
                candidates.append(idxs)
    return np.array(contexts, dtype = np.int32), np.array(candidates, dtype = np.int32)


FILE_PATHS = {"PTB": ("data/train.txt",
                      "data/valid.txt",
                      "data/test_blanks.txt",
                      "data/words.dict"),
            "PTB1000": ("data/train.1000.txt",
                      "data/valid.1000.txt",
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
    parser.add_argument('batch_size', help="Size of batch (n/b)", type=int)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    batch_size = args.batch_size
    train, valid, test, word_dict = FILE_PATHS[dataset]

    # Get word dict
    word_to_idx = get_vocab(word_dict)
    nclasses = len(word_to_idx.keys())

    # Convert data with n-gram windows
    train_input, train_output = convert_data(train, word_to_idx, batch_size)
    if valid:
        valid_input, valid_output = convert_data(valid, word_to_idx, batch_size)
    if test:
        test_input, test_candidates = convert_test_data(test, word_to_idx)

    filename = args.dataset + '_LSTM.hdf5'
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

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
