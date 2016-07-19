import numpy as np
import theano
import theano.tensor as T

from lstm import *
from updates import *

class WordEncoderLayer:
	def __init__(self, rng, X, in_size, out_size, hidden_size, cell, optimizer, drop_rate, is_train, mask):
		self.rng = rng
		self.X = X
		self.in_size = in_size
		self.out_size = out_size
		self.hidden_size_list = hidden_size
		self.cell = cell
		self.optimizer = optimizer
		self.drop_rate = drop_rate
		self.is_train = is_train
		self.mask = mask
		self.num_sents = mask.shape[1]

		self.define_layers()

	def define_layers(self):
		self.layers = []
		self.params = []

		for i in xrange(len(self.hidden_size_list)):
			if i == 0:
				layer_input = self.X
				shape = (self.in_size, self.hidden_size_list[0])
			else:
				layer_input = self.layers[i-1].activation
				shape = (self.hidden_size_list[i-1], self.hidden_size_list[i])

			hidden_layer = LSTMLayer(self.rng, str(i), shape, layer_input, self.mask, self.is_train, self.drop_rate, self.num_sents)

			self.layers.append(hidden_layer)
			self.params += hidden_layer.params

		self.activation = hidden_layer.activation
		self.hidden_size = hidden_layer.out_size