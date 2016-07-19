# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle, gzip
import numpy as np

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

def load_stop_words(f_path = None):
    stop_words = {}
    if f_path == None:
        f = open(curr_path + "/data/stopwords.txt", "r")
    else:
        f = open(curr_path + "/" + f_path, "r")
    for line in f:
        line = line.strip('\n').lower()
        stop_words[line] = 1

    return stop_words

###
# + Ignore line with length < 3 or separating line
# + Build term frequency when building vocabulary

###
def words_to_matrix(words, w2v_model, unknown_word_vector):
    x = []
    for j in range(0, len(words)):
        if words[j] in w2v_model:
            x.append(w2v_model[words[j]])
        else:
            x.append(unknown_word_vector)

    return np.asmatrix(x)


def word_sequence(f_path, w2v_model, unknown_word_vector, embedding_size):
    seqs = []
    i2w = {}
    w2i = {}
    lines = []
    tf = {}
    f = open(f_path, "r")

    part = 1

    content_sents = []
    summary_sents = []

    #ignore first 2 lines
    for _ in xrange(2):
        next(f)

    for line in f:
        line = line.strip('\n').lower()

        if len(line) == 0:
            if part == 1:
                part = 2
                continue
            else:
                break

        words = line.split()
        words.append("<eoss>") # end symbol
        if len(words) < 3 or line == "====":
            continue

        if part == 1:
            content_sents.append(words)
        else:
            summary_sents.append(words)
        #lines.append(words)
        for w in words:
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
                tf[w] = 1
            else:
                tf[w] += 1
    f.close()

    content_seqs = []
    summary_seqs = []

    for i in range(0, len(content_sents)):
        words = content_sents[i]
        content_seqs.append(words_to_matrix(words, w2v_model, unknown_word_vector))

    for i in range(0, len(summary_sents)):
        words = summary_sents[i]
        summary_seqs.append(words_to_matrix(words, w2v_model, unknown_word_vector))

    #padding the sequence to make sure it is the matrix
    data_x = pad_sequences(content_seqs, embedding_size)
    data_y = pad_sequences(summary_seqs, embedding_size)

    print "#dic = " + str(len(w2i))
    return data_x, data_y, w2v_model, unknown_word_vector

def pad_sequences(seqs, dim):
    zeros_m = np.zeros((1, dim), dtype = theano.config.floatX)
    max_len = 0

    for i in xrange(len(seqs)):
        if max_len < len(seqs[i]):
            max_len = len(seqs[i])

    mask = np.zeros((max_len, len(seqs)), dtype = theano.config.floatX)
    concat_X = np.zeros((max_len, len(seqs) * dim), dtype = theano.config.floatX)

    for i in xrange(len(seqs)):
        seq = seqs[i]
        X = seq[0: len(seq), ]
        mask[0:X.shape[0], i] = 1

        seq_len = len(seq)

        for r in xrange(max_len - seq_len):
            X = np.concatenate((X, zeros_m), axis=0)

        concat_X[:, i*dim: (i + 1)*dim] = X

    return [concat_X, mask]

if __name__ == "__main__":
    home = os.path.expanduser('~')
    data_x, data_y, i2w, w2i = word_sequence(home + "/research/data/summarization/dailymail/training/1134b173cd9ba85dea103bf1260357fa0619b08f.summary")
    print data_x[0].shape, data_x[1].shape
    print data_y[0].shape, data_y[1].shape
