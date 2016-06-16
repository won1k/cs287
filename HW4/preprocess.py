#!/usr/bin/env python

"""Part-Of-Speech Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

# Your preprocessing, features construction, and word2vec code.

def clean_char(char):
    curr, line, new = char.partition('\n')
    cleaned = False
    if line:
        cleaned = True
    return curr, line + ' ' + new + ' ', cleaned

def char_dict(file_list):
    char_to_idx = {}
    idx = 1
    nchars = 0
    for filename in file_list:
        if filename:
            with codecs.open(filename, 'r', encoding = 'latin-1') as f:
                remainder = ''
                while True:
                    char, space, remainder = remainder.partition(' ')
                    char, new, cleaned = clean_char(char)
                    if cleaned:
                        remainder = new + remainder
                    if space:
                        if char not in char_to_idx:
                            char_to_idx[char] = idx
                            idx += 1
                        nchars += 1
                    else:
                        next_chunk = f.read(1000)
                        if next_chunk:
                            remainder = remainder + next_chunk
                        else:
                            break
    return char_to_idx, nchars + 1

def convert_data(filename, char_to_idx, nchars, seqlen, bsize):
    nmatrix = bsize / seqlen # number of matrices per row
    char_matrix = [[] for i in range(nmatrix)]
    spaces = []
    print nmatrix

    # Add padding to ensure bsize | nchars
    padding = bsize - (nchars % bsize)
    if padding == bsize:
        padding_needed = False
    else:
        padding_needed = True

    with codecs.open(filename, 'r', encoding = 'latin-1') as f:
        remainder = f.read(1000)
        curr_batch = 0
        mat_row = []
        while True:
            char, space, remainder = remainder.partition(' ')
            if space: # i.e. char or <space> found
                if char == '<space>':
                    spaces.append(2)
                else:
                    spaces.append(1)
                if curr_batch < bsize: # i.e. still on same batch (row)
                    if len(mat_row) < seqlen: # i.e. still in same matrix
                        mat_row.append(char_to_idx[char])
                        curr_batch += 1
                    else: # filled matrix row in last iteration
                        char_matrix[curr_batch / seqlen - 1].append(mat_row)
                        mat_row = [char_to_idx[char]]
                        curr_batch += 1
                else: # last iteration filled full row of all matrices
                    char_matrix[curr_batch / seqlen - 1].append(mat_row)
                    mat_row = [char_to_idx[char]]
                    curr_batch = 1
            else:
                next_chunk = f.read(1000)
                if next_chunk:
                    remainder = remainder + next_chunk
                else:
                    if padding_needed:
                        remainder = remainder + ' </s>' * padding
                        padding_needed = False
                    break
    for matrix in char_matrix:
        for row in matrix:
            if len(row) != seqlen:
                print len(row)
    return np.array(char_matrix), np.array(spaces, dtype = np.int32)



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
    args = parser.parse_args(arguments)
    dataset = args.dataset
    seqlen = args.seqlen
    bsize = args.batch_size
    train, valid, test = FILE_PATHS[dataset]

    # Get char dict
    char_to_idx, nchars = char_dict([train, valid, test])

    # Convert data with n-gram windows
    train_input, train_output = convert_data(train, char_to_idx, nchars, seqlen, bsize)
    if valid:
        valid_input, valid_output = convert_data(valid, char_to_idx, nchars, seqlen, bsize)
    if test:
        test_input, test_candidates = convert_test_data(test, char_to_idx)

    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
        f['nfeatures'] = np.array([V], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
