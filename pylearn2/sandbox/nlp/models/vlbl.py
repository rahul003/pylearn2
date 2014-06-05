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
from pylearn2.sandbox.nlp.models.lblcost import Default
#import ipdb



class vLBL(Model):

    def __init__(self, dict_size, dim, context_length, k, irange = 0.1, seed = 22):
        #dim is the dimensions of the final representation of each word

        super(vLBL, self).__init__()
        rng = np.random.RandomState(seed)
        self.rng = rng
        self.k = k
        self.context_length = context_length
        self.dim = dim
        self.dict_size = dict_size
        C = rng.randn(dim, context_length)
        self.C = sharedX(C) 

        #right now i think it is both in same prepresentation. meaning R=Q=W
        W = rng.uniform(-irange, irange, (dict_size, dim))
        W = sharedX(W,name='W')
        self.projector = MatrixMul(W)

        self.b = sharedX(np.zeros((dict_size,)), name = 'vLBL_b')

        self.input_space = IndexSpace(dim = context_length, max_labels = dict_size)
        self.output_space = IndexSpace(dim = 1, max_labels = dict_size)

        self.allY = T.as_tensor_variable(np.arange(dict_size,dtype=np.int64).reshape(dict_size,1))


    def get_params(self):
        #get W from projector
        rval = self.projector.get_params()
        #add C, b
        rval.extend([self.C, self.b])
        return rval


    def fprop(self, state_below):
        """
        state_below is r_w?
        """
        state_below = state_below.reshape((state_below.shape[0], self.dim, self.context_length))
        rval = self.C.dimshuffle('x', 0, 1) * state_below
        rval = rval.sum(axis=2)
        return rval


    def score(self, X, Y):
        """
        So takes X which is context.
        gets r_w which project returns
        gets q_hat = from calculating fprop which gives us the predicted representation.
        Then finds score by doing dot product with the target representation
        """

        X = self.projector.project(X)
        q_h = self.fprop(X)

        q_w = self.projector.project(Y)
        # print Y.dtype
        # print self.allY.dtype
        # print Y.type
        # print type(Y)
        # print type(self.allY)
        all_q_w = self.projector.project(self.allY)
        
        swh = (q_w*q_h).sum(axis=1) + self.b[Y].flatten()
        sallwh = T.dot(q_h,all_q_w.T) + self.b.dimshuffle('x',0)
        
        swh = T.exp(swh)
        sallwh = T.exp(sallwh).sum(axis=1)

        return swh,sallwh
        #10,5
    
        #dim is n_examples x n_dim_word_representation

        # if ndim == 1: 
        #     #for vector this is the case
        #         #.reshape((Y.shape[0], self.dim))
        #     q_w = self.projector.project(Y)
        #     rval = (q_w * q_h).sum(axis=1) + self.b[Y].flatten()
        # elif ndim == 2:
        #     #q_w = self.projector.project(Y).reshape((Y.shape[0], Y.shape[1], self.dim)).dimshuffle(1, 0, 2)
        #     #rval = (q_h.dimshuffle('x', 0, 1) + q_w).sum(axis=1) + self.b[Y].flatten()
        #     rval = (q_h.dimshuffle('x', 0, 1) * q_w).sum(axis=2) + self.b[Y].flatten()
        #return rval

    # def delta(self, data, ndim = 1):

    #     X, Y = data
    #     p_n = 1. / self.dict_size

    #     #return self.score(X, Y) - T.log(self.k * p_n[Y])
    #     return self.score(X, Y, ndim = ndim) - T.log(self.k * p_n)

    def get_default_cost(self):
        return Default()

    def cost_from_X(self, data):
        X, Y = data
        theano_rng = RandomStreams(seed = self.rng.randint(2 ** 15))

        s,denom = self.score(X,Y)
        
        p_w_given_h = s/denom
        #15x1
        
        #T.arange(Y.shape[0]), Y])
        return -T.mean(T.log2(p_w_given_h))
        
        #noise = theano_rng.random_integers(size = (X.shape[0], self.k,), low=0, high = self.dict_size - 1)

        #pos = T.log(self.delta(data, ndim = 1))
        #neg = T.log(self.delta((X, noise), ndim = 2))

        #import ipdb
        #ipdb.set_trace()
        #return pos - neg.sum()
