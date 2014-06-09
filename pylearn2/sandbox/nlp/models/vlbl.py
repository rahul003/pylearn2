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
#from theano.sandbox.rng_mrg import MRG_RandomStreams
from pylearn2.utils import sharedX
from pylearn2.sandbox.nlp.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.sandbox.nlp.models.lblcost import Default
#import ipdb
from pylearn2.space import CompositeSpace


class vLBL(Model):

    def __init__(self, dict_size, dim, context_length, k, irange = 0.1, seed = 22):
        #dim is the dimensions of the final representation of each word

        super(vLBL, self).__init__()
        rng = np.random.RandomState(seed)
        self.rng = rng
        self.context_length = context_length
        self.dim = dim
        self.dict_size = dict_size
        C = rng.randn(dim, context_length)
        self.C = sharedX(C) 

        #right now i think it is both in same prepresentation. meaning R=Q=W
        W_context = rng.uniform(-irange, irange, (dict_size, dim))
        W_context = sharedX(W_context,name='W_context')
        W_target = rng.uniform(-irange, irange, (dict_size, dim))
        W_target = sharedX(W_target,name='W_target')
        self.projector_context = MatrixMul(W_context)
        self.projector_target = MatrixMul(W_target)
        
        self.W_context = W_context
        self.W_target = W_target

        self.b = sharedX(np.zeros((dict_size,)), name = 'vLBL_b')

        self.input_space = IndexSpace(dim = context_length, max_labels = dict_size)
        self.output_space = IndexSpace(dim = 1, max_labels = dict_size)

        self.allY = T.as_tensor_variable(np.arange(dict_size,dtype=np.int64).reshape(dict_size,1))


    def get_params(self):
        #get W from projector
        rval1 = self.projector_context.get_params()
        rval2 = self.projector_target.get_params()
                #add C, b
        rval1.extend([self.C, self.b])
        rval1.extend(rval2)
        return rval1


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

        X = self.projector_context.project(X)
        q_h = self.fprop(X)

        q_w = self.projector_target.project(Y)
        all_q_w = self.projector_target.project(self.allY)
        
        #swh = (q_w*q_h).sum(axis=1) + self.b[Y].flatten()
        sallwh = T.dot(q_h,all_q_w.T) + self.b.dimshuffle('x',0)
        #swh = T.exp(swh)
        #sallwh = T.exp(sallwh).sum(axis=1)

        #return s, sallwh
        return sallwh



        
        #10,5

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

    def get_default_cost(self):
        return Default()

    def get_monitoring_data_specs(self):
        """
        Returns data specs requiring both inputs and targets.

        Returns
        -------
        data_specs: TODO
            The data specifications for both inputs and targets.
        """
        space = CompositeSpace((self.get_input_space(),
                                self.get_output_space()))
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)

    def get_monitoring_channels(self, data):
        #if(data is None):
          #  return
        W_context = self.W_context
        W_target = self.W_target
        b = self.b
        C = self.C

        sq_W_context = T.sqr(W_context)
        sq_W_target = T.sqr(W_target)
        sq_b = T.sqr(b)
        sq_c = T.sqr(C)

        row_norms_W_context = T.sqrt(sq_W_context.sum(axis=1))
        col_norms_W_context = T.sqrt(sq_W_context.sum(axis=0))

        row_norms_W_target = T.sqrt(sq_W_target.sum(axis=1))
        col_norms_W_target = T.sqrt(sq_W_target.sum(axis=0))
        
        col_norms_b = T.sqrt(sq_b.sum(axis=0))

        
        col_norms_c = T.sqrt(sq_c.sum(axis=0))

        rval = OrderedDict([
                            ('W_context_row_norms_min'  , row_norms_W_context.min()),
                            ('W_context_row_norms_mean' , row_norms_W_context.mean()),
                            ('W_context_row_norms_max'  , row_norms_W_context.max()),
                            ('W_context_col_norms_min'  , col_norms_W_context.min()),
                            ('W_context_col_norms_mean' , col_norms_W_context.mean()),
                            ('W_context_col_norms_max'  , col_norms_W_context.max()),

                            ('W_target_row_norms_min'  , row_norms_W_target.min()),
                            ('W_target_row_norms_mean' , row_norms_W_target.mean()),
                            ('W_target_row_norms_max'  , row_norms_W_target.max()),
                            ('W_target_col_norms_min'  , col_norms_W_target.min()),
                            ('W_target_col_norms_mean' , col_norms_W_target.mean()),
                            ('W_target_col_norms_max'  , col_norms_W_target.max()),
                            
                            ('b_col_norms_min'  , col_norms_b.min()),
                            ('b_col_norms_mean' , col_norms_b.mean()),
                            ('b_col_norms_max'  , col_norms_b.max()),

                            ('c_col_norms_min'  , col_norms_c.min()),
                            ('c_col_norms_mean' , col_norms_c.mean()),
                            ('c_col_norms_max'  , col_norms_c.max()),
                            ])

        rval['nll'] = self.cost_from_X(data)
        rval['perplexity'] = 10 ** (rval['nll']/np.log(10).astype('float32'))
        return rval
        
    def cost_from_X(self, data):
        X, Y = data
        s = self.score(X,Y)
        p_w_given_h = T.nnet.softmax(s)
        #15x1
        #T.arange(Y.shape[0]), Y])
        return -T.mean(T.log2(p_w_given_h)[T.arange(Y.shape[0]), Y])

class vLBLNCE(vLBL):
    
    def __init__(self, dict_size, dim, context_length, k, irange = 0.1, seed = 22):
        super(vLBLNCE, self).__init__(dict_size, dim, context_length,k)
        self.k = k
    def score(self, X, Y):
        """
        So takes X which is context.
        gets r_w which project returns
        gets q_hat = from calculating fprop which gives us the predicted representation.
        Then finds score by doing dot product with the target representation
        """

        X = self.projector_context.project(X)
        q_h = self.fprop(X)

        q_w = self.projector_target.project(Y)
        # print Y.dtype
        # print self.allY.dtype
        # print Y.type
        # print type(Y)
        # print type(self.allY)
        #all_q_w = self.projector_target.project(self.allY)
        
        swh = (q_w*q_h).sum(axis=1) + self.b[Y].flatten()
        #sallwh = T.dot(q_h,all_q_w.T) + self.b.dimshuffle('x',0)
        #swh = T.exp(swh)
        #sallwh = T.exp(sallwh).sum(axis=1)
        return swh

    #10,5
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

    
    def delta(self, data, ndim = 1):
        X, Y = data
        p_n = 1. / self.dict_size
        de = self.score(X,Y)
        de = de - T.log(self.k*p_n)
        #this is only for uniform(?)
        return de

        #return self.score(X, Y, ndim = ndim) - T.log(self.k * p_n)
    def prob_data_given_word_theta(self,delta_rval):
        return T.nnet.sigmoid(delta_rval)

    def cost_from_X(self,data):
       
        delta_rval = self.delta(data)
        prob = self.prob_data_given_word_theta(delta_rval)
        logprob = T.log(prob)
        logprobnoise = T.log(1-prob) 
        
        return -(T.mean(logprob)+T.mean(logprobnoise))
        #return -T.mean(delta_rv)
        #expectation over data of log prob_data_given_word_theta
        # + k* (expectation over noise distribution)[1 - prob_data_given_word_theta]
        
