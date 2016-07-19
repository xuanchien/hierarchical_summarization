import numpy as np
import theano
import theano.tensor as T
from word_encoder import *
from sent_encoder import *
from sent_decoder import *
from word_decoder import *
from updates import *

class RNN:
	def __init__(self, in_size, out_size, hidden_size, cell = 'LSTM', optimizer = 'rmsprop', drop_rate = 0.5, num_output_sents = 1):
		self.X = T.matrix('X')
		self.Y = T.matrix('Y')
		self.mask_X = T.matrix('mask_X')
		self.mask_Y = T.matrix('mask_Y')

		self.in_size = in_size
		self.out_size = out_size
		self.hidden_size = hidden_size
		self.cell = cell
		self.drop_rate = drop_rate
		self.is_train = T.iscalar('is_train')
		self.optimizer = optimizer

		self.define_layers()
		self.define_train_test_funcs()

	def define_layers(self):
		self.layers = []
		self.params = []

		random_generator = np.random.RandomState(1234)

		# Word Encoder
		word_encoder_layer = WordEncoderLayer(random_generator, self.X, self.in_size, self.out_size, self.hidden_size, self.cell, self.optimizer, self.drop_rate, self.is_train, self.mask_X)
		self.layers += word_encoder_layer.layers
		self.params = word_encoder_layer.params

		i = len(self.layers) - 1

		# Sentence Encoder
		layer_input = word_encoder_layer.activation
		sent_encoder_layer = SentEncoderLayer(self.cell, random_generator, str(i+1), (word_encoder_layer.hidden_size, word_encoder_layer.hidden_size), layer_input, self.mask_X, self.is_train, self.drop_rate)

		self.layers.append(sent_encoder_layer)
		self.params += sent_encoder_layer.params

		# Sentence Decoder
		codes = sent_encoder_layer.activation
		codes = T.reshape(codes, (1, sent_encoder_layer.out_size))

		sent_decoder_layer = SentDecoderLayer(self.cell, random_generator, str(i+2), (sent_encoder_layer.out_size, sent_encoder_layer.in_size), codes, self.mask_Y, self.is_train, self.drop_rate)
		self.layers.append(sent_decoder_layer)
		self.params += sent_decoder_layer.params

		# Word Decoder
		sent_codes = sent_decoder_layer.activation
		sents_codes = T.reshape(sents_codes, (1, self.mask_Y.shape[1] * sent_decoder_layer.out_size))
		word_decoder_layer = WordDecoderLayer(self.cell, random_generator, str(i+3), (sent_decoder_layer.out_size, self.out_size), sent_codes, self.mask_Y, self.is_train, self.drop_rate)
		self.layers.append(word_decoder_layer)
		self.params += word_decoder_layer.params
		self.activation = word_decoder_layer.activation

		self.epsilon = 1.0e-15

	def categorical_crossentropy(self, y_pred, y_true):
		y_pred = T.clip(y_pred, self.epsilon, 1.0 - self.epsilon)

		m = T.reshape(self.mask_Y, (self.mask_Y.shape[0] * self.mask_Y.shape[1], 1))

		ce = T.nnet.categorical_crossentropy(y_pred, y_true)
		ce = T.reshape(ce, (self.mask_Y.shape[0] * self.mask_Y.shape[1], 1))

		return T.sum(ce * m) / T.sum(m)

	def define_train_test_funcs(self):
		num_output_sents = self.mask_Y.shape[1]
		pYs = T.reshape(self.activation, (self.mask_Y.shape[0] * num_output_sents, self.out_size))

		tYs = T.reshape(self.Y, (self.mask_Y.shape[0] * num_output_sents, self.out_size))

		cost = self.categorical_crossentropy(pYs, tYs)

		gparams = []
		for param in self.params:
			gparam = T.grad(cost, param)
			gparams.append(gparam)

		lr = T.scalar("lr") #learning rate

		optimizer = eval(self.optimizer)
		updates = optimizer(self.params, gparams, lr)

		self.train = theano.function(inputs = [self.X, self.mask_X, self.Y, self.mask_Y, lr], givens = {self.is_train: np.cast['int32'](1)}, outputs = [cost, self.activation], updates = updates)
