import theano
import theano.tensor as T

X = T.matrix('X')
y = X.shape[1] + 10

func_sum = theano.function([X], y)

func_sum([[0, 1], [1, 2], [2, 3]])