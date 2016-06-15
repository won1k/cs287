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

def char_dict(file_list):
    char_to_idx = {}
    idx = 1
    for filename in file_list:
        if filename:
            with codecs.open(filename, 'r', encoding = 'latin-1') as f:
                remainder = ''
                while True:
                    char, space, remainder = remainder.partition(' ')
                    if space:
                        # char or space was found
                        if char != '<space>':
                            if char not in char_to_idx:
                                char_to_idx[char] = idx
                                idx += 1
                                if not char.isalpha():
                                    print char
                                    print filename
                    else:
                        next_chunk = f.read(1000)
                        if next_chunk:
                            remainder = remainder + next_chunk
                        else:
                            break
    return char_to_idx

def convert_data(filename, char_to_idx, seqlen, nbatch):
    with codecs.open(filename, 'r', encoding = 'latin-1') as f:
        
    return



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
    parser.add_argument('nbatch', help="Number of batches b (for n/b total rows)", type=int)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    seqlen = args.seqlen
    nbatch = args.nbatch
    train, valid, test = FILE_PATHS[dataset]

    # Get char dict
    char_to_idx = char_dict([train, valid, test])
    nchars = len(char_to_idx.keys())

    # Convert data with n-gram windows
    train_input, train_output = convert_data(train, char_to_idx, seqlen, nbatch)
    if valid:
        valid_input, valid_output = convert_data(valid, char_to_idx)
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
