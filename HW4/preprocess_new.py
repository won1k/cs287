#!/usr/bin/env python

"""Part-Of-Speech Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

SPACE = '<space>'
EOF = '</s>'

# Your preprocessing, features construction, and word2vec code.

def clean_char(char):
    curr, line, new = char.partition('\n')
    cleaned = False
    if line:
        cleaned = True
    return curr, line + ' ' + new + ' ', cleaned

def char_dict(file_list, spaces):
    char_to_idx = {}
    idx = 1
    for filename in file_list:
        if filename:
            with codecs.open(filename, 'r', encoding = 'latin-1') as f:
                text = f.readline()
                chars = text.split(' ')
                for char in chars:
                    if spaces == 0:
                        if char == SPACE:
                            continue
                    if char not in char_to_idx:
                        char_to_idx[char] = idx
                        idx += 1
    return char_to_idx

def convert_data(filename, char_to_idx, bsize, spaces):
    char_matrix = []
    space_matrix = []
    with codecs.open(filename, 'r', encoding = 'latin-1') as f:
        chars = f.readline().split(' ')
        char_row = []
        space_row = []
        for idx, char in enumerate(chars):
            if spaces == 0:
                if char == SPACE:
                    continue

            char_row.append(char_to_idx[char])
            try:
                space_row.append(2 if chars[idx + 1] == SPACE else 1)
            except:
                space_row.append(1)

            if len(char_row) == bsize:
                char_matrix.append(char_row)
                char_row = []
                space_matrix.append(space_row)
                space_row = []

        curr_len = len(char_row)
        if curr_len > 0 and curr_len < bsize: # i.e. need padding
            char_row += [char_to_idx[EOF]] * (bsize - curr_len)
            char_matrix.append(char_row)
            space_row += [1] * (bsize - curr_len)
            space_matrix.append(space_row)
        #padding = (bsize - (len(chars) % bsize)) % bsize
        #chars += ['</s>'] * padding
        #nrows = len(chars) / bsize
        #for i in range(nrows):
        #    row = chars[(bsize * i):(bsize * (i+1))]
        #    char_matrix.append([char_to_idx[char] for char in row])
        #    spaces.append([2 if char == '<space>' else 1 for char in row])
    return np.array(char_matrix, dtype = np.int32), np.array(space_matrix, dtype = np.int32)

FILE_PATHS = {"PTB": ("data/train_chars.txt",
                      "data/valid_chars.txt",
                      "data/test_chars.txt")}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    parser.add_argument('seqlen', help="Sequence length for backprop", type=int)
    parser.add_argument('batch_size', help="Size of batch (n/b)", type=int)
    parser.add_argument('spaces', help="Use space in train/valid (1 if true, 0 if false)", type=int)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    seqlen = args.seqlen
    bsize = args.batch_size
    spaces = args.spaces
    train, valid, test = FILE_PATHS[dataset]

    # Get char dict
    char_to_idx = char_dict([train, valid, test], spaces)
    nfeatures = len(char_to_idx.keys())

    # Convert data with n-gram windows
    train_input, train_output = convert_data(train, char_to_idx, bsize, spaces)
    if valid:
        valid_input, valid_output = convert_data(valid, char_to_idx, bsize, spaces)
    if test:
        test_input, test_output = convert_data(test, char_to_idx, bsize, spaces)

    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
        f['nfeatures'] = np.array([nfeatures], dtype=np.int32)
        f['seqlen'] = np.array([seqlen], dtype=np.int32)
        f['bsize'] = np.array([bsize], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
