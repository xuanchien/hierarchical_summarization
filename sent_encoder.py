import numpy as np
import theano
import theano.tensor as T

from lstm import *
from updates import *

class SentEncoderLayer:
	def __init__(self, cell, rng, layer_id, shape, X, mask, is_train = 1, p = 0.5):
		prefix = "SentEncoder_"
		self.in_size, self.out_size = shape
		num_sents = mask.shape[1]

		# retrieve the last row of the input matrix X
		# this is the output of the LSTM on each sentence
		sent_X = T.reshape(X[X.shape[0] - 1, :], (num_sents, self.in_size))

		# create a mask matrix, size = (num_sents x 1)
		mask = T.reshape(T.ones_like(sent_X)[:, 0], (num_sents, 1))

		# we will consider this as a sentence where each token is the last vector output from the WordEncoder layer
		# then, we apply LSTM on this sentence as usual (notice that the last param is 1)
		self.encoder = LSTMLayer(rng, prefix + layer_id, shape, sent_X, mask, is_train, p, 1)

		#activation is the last vector returned by the LSTM
		self.activation = self.encoder.activation[self.encoder.activation.shape[0] - 1, :]

		self.sent_encs = sent_X
		self.params = self.encoder.params