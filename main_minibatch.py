import os
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from rnn import *
import data
from utils import *
from gensim.models import Word2Vec

use_gpu(1) # -1:cpu; 0,1,2,..: gpu

print("Loading w2v model")
w2v_model = Word2Vec.load_word2vec_format(os.path.expanduser("~") + "/sskip.100.vectors", binary=False)

e = 0.01
lr = 0.1 #learning rate
drop_rate = 0.
batch_size = 1
hidden_size = [500]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
optimizer = "nesterov_momentum"
embedding_size = 100
unknown_word_vector = np.random.uniform(-0.5, 0.5, embedding_size)
epoch = 10

# data_xy contains list of batches, each batch has size = batch_size
training_folder = sys.argv[1]

files = os.listdir(training_folder)
files = [file for file in files if not file.startswith(".")]
number_of_batch = len(files) / batch_size

print "initializing model"
model = RNN(embedding_size, embedding_size, hidden_size)

for _ in xrange(epoch):
  print "new epoch"
  error = 0.0
  in_start = time.time()
  for i in xrange(number_of_batch):
    print "batch: ", i
    training_files = files[i*batch_size : (i+1)*batch_size]

    for file in training_files:
      print "loading data from file: ", file
      data_x, data_y, w2v_model, unknown_word_vector = data.word_sequence(os.path.join(training_folder, file),
        w2v_model,
        unknown_word_vector,
        embedding_size)

      X, mask_X = data_x
      Y, mask_Y = data_y

      print "training"
      cost, sents = model.train(X, mask_X, Y, mask_Y, lr)
      error += cost

      print "decoding to text"
      for s in xrange(int(sents.shape[1] / embedding_size)):
        xs = sents[:, s * embedding_size : (s + 1) * embedding_size]
        for w_i in xrange(xs.shape[0]):
          vector = xs[w_i, :]
          w, score = w2v_model.similar_by_vector(vector, topn=1)[0]
          unknown_score = consine_similarity(vector, unknown_word_vector)

          if (score < unknown_score):
            w = '<UNK>'
          print w,
        print "\n"

  in_time = time.time() - in_start





