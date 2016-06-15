import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import csv
import operator
import preprocess

from sklearn.neighbors import NearestNeighbors

# Global preprocessing variables
unk = 2
start = 3
end = 4

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
    parser.add_argument('metric', help='KNN metric', type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test, word_dict = FILE_PATHS[dataset]
    metric = args.metric

    # Get word dict
    word_to_idx = preprocess.get_vocab(word_dict)
    idx_to_word = dict((idx,word) for word,idx in word_to_idx.iteritems())
    nclasses = len(word_to_idx.keys())

    # Open filename
    f = h5py.File('LT_weights.hdf5','r')
    ltweights = f['weights']

    # Get most similar words
    print(ltweights)
    nbrs = NearestNeighbors(10, 1, algorithm = 'brute', metric = metric)
    nbrs.fit(ltweights)
    distances, indices = nbrs.kneighbors(ltweights)
    min_dist = [dist[1] for dist in distances]
    sorted_min_indices = np.argsort(min_dist)[:10] # pick top 10 most similar words
    for i in range(10):
        print idx_to_word[sorted_min_indices[i] + 1] + " : " + idx_to_word[indices[sorted_min_indices[i]][1]]

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
