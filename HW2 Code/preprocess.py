#!/usr/bin/env python

"""Part-Of-Speech Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import csv
import more_itertools as mi

# Global preprocessing variables
s = 'PADDING'
rare = 'RARE'
num = 'NUMBER'
N = 100000 # number of most common words to keep

# Your preprocessing, features construction, and word2vec code.

def clean_word(word):
    """
    Clean for number, capitalization.
    """
    cap = 2 # no remarkable feature (1 is padding)
    if word.islower(): # i.e. all lower
        cap = 3
    elif word.isupper(): # all caps
        cap = 4
    else:
        if word[0].isupper(): # first letter capital
            cap = 5
        elif len([l for l in word if l.isupper()]) == 1: # exactly one capital, not first
            cap = 6

    word = word.lower()
    word = re.sub('(\d+)', 'NUMBER', word)
    return word, cap


def get_vocab(file_list):
    """
    Construct index feature dictionary.
    Keep only 100,000 most common words.
    """
    word_to_idx = {}
    word_counts = {}

    # Get ranking of word counts
    for filename in file_list:
        if filename:
            with open(filename, "r") as f:
                f = csv.reader(f, delimiter = '\t')
                for row in f:
                    if row:
                        word, cap = clean_word(row[2])
                        if word not in word_counts:
                            word_counts[word] = 1
                        else:
                            word_counts[word] += 1
    # Keep only top 100,000 common words
    idx = 3 # 1 is padding, 2 is rare
    word_to_idx[s] = 1
    word_to_idx[rare] = 2
    for word in sorted(word_counts):
        if idx - 2 <= N:
            word_to_idx[word] = idx
            idx += 1
    return word_to_idx

def get_vocab_pre():
    """
    Construct word feature dictionary using 400,000 words in pretrained features.
    Output both word_to_idx and matrix of feature embeddings.
    """
    word_to_idx = {}
    word_to_idx[s] = 1
    word_to_idx[rare] = 2
    pretrained_features = []
    pretrained_features.append(np.random.randn(50)) # padding; becomes index 1 in Torch
    pretrained_features.append(np.random.randn(50)) # rare; becomes index 2 in Torch
    idx = 3
    with open("data/glove.6B.50d.txt","r") as f:
        f = csv.reader(f, delimiter = ' ', quoting=csv.QUOTE_NONE)
        for row in f:
            word_to_idx[row[0]] = idx
            pretrained_features.append([float(x) for x in row[1:]])
            idx += 1
    return word_to_idx, np.array(pretrained_features, dtype = np.float64)


def convert_data(data_name, word_to_idx, tags_dict, dwin):
    """
    Convert data to windowed word/cap indices.
    """
    with open(data_name,'r') as f: # row = [global_id, sent_id, word, TAG]
        # Get number of rows; initialize
        n = len([1 for row in csv.reader(f, delimiter = '\t') if row])
        word_windows = [[] for i in range(n)]
        cap_windows = [[] for i in range(n)]
        labels = []

        # Return to start of file; use peekable, csv reader
        f.seek(0)
        f = csv.reader(f, delimiter = '\t')
        f = mi.peekable(f) # allow for "peeking" next row to check if end of sentence

        for row in f:
            # Check if blank row
            if not row:
                continue

            # Add labels (test doesn't have labels)
            try:
                labels.append(tags_dict[row[3]])
            except:
                pass

            # Clean word to proper form (i.e. psNUMBER); initialize variables
            word, cap = clean_word(row[2])
            global_id = int(row[0])
            sent_id = int(row[1])

            # Get word index
            try:
                word_idx = word_to_idx[word]
            except:
                word_idx = word_to_idx[rare]

            # Initialize for start of sentence
            if sent_id == 1:
                for i in range(dwin/2 + 1):
                    word_windows[global_id + i - 1] = [word_to_idx[s]] * (dwin/2 - i) + [word_idx]
                    cap_windows[global_id + i - 1] = [1] * (dwin/2 - i) + [cap] # padding "caps" feature is N + 2 + 1
            # If not start, append embedded word to each appropriate row
            else:
                for i in range(-int(dwin/2), dwin/2 + 1):
                    # Check that after start of sentence (before end doesn't matter since rewritten)
                    if sent_id + i > 0:
                        try: # may be end of file
                            word_windows[global_id + i - 1].append(word_idx)
                            cap_windows[global_id + i - 1].append(cap)
                        except:
                            print "End of file"
            # Also check if end of sentence; add appropriate padding
            if f.peek() == []:
                for i in range(-int(dwin/2), 1):
                    if sent_id + i > 0:
                        word_windows[global_id + i - 1] += [word_to_idx[s]] * (dwin/2 + i)
                        cap_windows[global_id + i - 1] += [1] * (dwin/2 + i)
    #return word_windows
    return np.array(word_windows, dtype = np.float64), np.array(cap_windows, dtype = np.float64), np.array(labels, dtype = np.int32)




FILE_PATHS = {"PTB": ("data/train.tags.txt",
                      "data/dev.tags.txt",
                      "data/test.tags.txt",
                      "data/tags.dict")}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    parser.add_argument('window_size', help = "Window size", type=int)
    parser.add_argument('pretrained', help = "Pretrained features (True/false)", type=bool)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    dwin = int(args.window_size)
    pretrained = args.pretrained
    train, valid, test, tag_dict = FILE_PATHS[dataset]

    # Open tag_dict as dictionary {'tag': idx}
    tags_dict = {}
    with open(tag_dict, 'r') as f:
        f = csv.reader(f, delimiter = '\t')
        for row in f:
            tags_dict[row[0]] = int(row[1])
    nclasses = len(tags_dict)

    # Get word dict
    if pretrained:
        word_to_idx, pretrained_features = get_vocab_pre()
    else:
        word_to_idx = get_vocab([train, valid, test])
    nfeatures = N + 2 + 6 # 2 for padding/rare, 6 (padding + none + 4 features) for caps

    # Dataset name
    train_input_word_windows, train_input_cap_windows, train_output = convert_data(train, word_to_idx, tags_dict, dwin)
    if valid:
        valid_input_word_windows, valid_input_cap_windows, valid_output = convert_data(valid, word_to_idx, tags_dict, dwin)
    if test:
        test_input_word_windows, test_input_cap_windows, test_output = convert_data(test, word_to_idx, tags_dict, dwin)

    # Output data
    filename = args.dataset + '55' + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input_word_windows'] = train_input_word_windows
        f['train_input_cap_windows'] = train_input_cap_windows
        f['train_output'] = train_output
        if valid:
            f['valid_input_word_windows'] = valid_input_word_windows
            f['valid_input_cap_windows'] = valid_input_cap_windows
            f['valid_output'] = valid_output
        if test:
            f['test_input_word_windows'] = test_input_word_windows
            f['test_input_cap_windows'] = test_input_cap_windows
        if pretrained:
            f['pretrained_features'] = pretrained_features
        f['nfeatures'] = np.array([nfeatures], dtype=np.int32)
        f['nclasses'] = np.array([nclasses], dtype=np.int32)
        f['dwin'] = np.array([dwin], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
