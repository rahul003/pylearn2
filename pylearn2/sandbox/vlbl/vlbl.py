import numpy as np
from theano import tensor as T
from theano import config
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
from theano.compat.python2x import OrderedDict
from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Softmax
from pylearn2.monitor import get_monitor_doc
from pylearn2.space import VectorSpace, IndexSpace
from pylearn2.format.target_format import OneHotFormatter
from theano.sandbox.rng_mrg import MRG_RandomStreams
from pylearn2.utils import sharedX
from pylearn2.sandbox.nlp.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
#import ipdb



class vLBL(Model):

    def __init__(self, dict_size, dim, context_length, k, irange = 0.1, seed = 22):

        rng = np.random.RandomState(seed)
        self.rng = rng
        self.k = k
        self.context_length = context_length
        self.dim = dim
        self.dict_size = dict_size
        C = rng.randn(dim, context_length)
        self.C = sharedX(C)

        W = rng.uniform(-irange, irange, (dict_size, dim))
        W = sharedX(W)
        self.projector = MatrixMul(W)

        self.b = sharedX(np.zeros((dict_size,)), name = 'vLBL_b')

        self.input_space = IndexSpace(dim = context_length, max_labels = dict_size)
        self.output_space = IndexSpace(dim = 1, max_labels = dict_size)

    def get_params(self):

        rval = self.projector.get_params()
        rval.extend([self.C, self.b])
        return rval


    def fprop(self, state_below):

        state_below = state_below.reshape((state_below.shape[0], self.dim, self.context_length))
        rval = self.C.dimshuffle('x', 0, 1) * state_below
        rval = rval.sum(axis=2)

        return rval


    def score(self, X, Y, ndim = 1):
        X = self.projector.project(X)
        q_h = self.fprop(X)
        if ndim == 1:
            q_w = self.projector.project(Y).reshape((Y.shape[0], self.dim))
            rval = (q_w + q_h).sum(axis=1) + self.b[Y].flatten()
        elif ndim == 2:
            q_w = self.projector.project(Y).reshape((Y.shape[0], Y.shape[1], self.dim)).dimshuffle(1, 0, 2)
            rval = (q_h.dimshuffle('x', 0, 1) + q_w).sum(axis=1) #+ self.b[Y].flatten()

        return rval


    def delta(self, data, ndim = 1):

        X, Y = data
        p_n = 1. / self.dict_size

        #return self.score(X, Y) - T.log(self.k * p_n[Y])
        return self.score(X, Y, ndim = ndim) - T.log(self.k * p_n)


    def cost_from_X(self, data):
        X, Y = data
        theano_rng = RandomStreams(seed = self.rng.randint(2 ** 15))
        noise = theano_rng.random_integers(size = (X.shape[0], self.k,), low=0, high = self.dict_size - 1)

        pos = T.log(self.delta(data, ndim = 1))
        neg = T.log(self.delta((X, noise), ndim = 2))

        import ipdb
        ipdb.set_trace()
        return pos - neg.sum()
