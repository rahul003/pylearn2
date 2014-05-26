"""
Sandbox multilayer perceptron layers for natural language processing (NLP)
"""
import theano.tensor as T
from theano import config
from theano.gof.op import get_debug_values
from theano.printing import Print
import theano
from pylearn2.models import mlp
from pylearn2.models.mlp import Layer
from pylearn2.space import IndexSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.sandbox.nlp.linear.matrixmul import MatrixMul
from theano.compat.python2x import OrderedDict
from pylearn2.utils import logger
from pylearn2.format.target_format import OneHotFormatter
from pylearn2.monitor import get_monitor_doc
import numpy as np

from pylearn2.utils import serial

class Softmax(mlp.Softmax):
    """
    An extension of the MLP's softmax layer which monitors
    the perplexity

    Parameters
    ----------
    n_classes : WRITEME
    layer_name : WRITEME
    irange : WRITEME
    istdev : WRITEME
    sparse_init : WRITEME
    W_lr_scale : WRITEME
    b_lr_scale : WRITEME
    max_row_norm : WRITEME
    no_affine : WRITEME
    max_col_norm : WRITEME
    init_bias_target_marginals : WRITEME
    """
    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):

        mx = state.max(axis=1)

        rval = OrderedDict([('mean_max_class', mx.mean()),
                            ('max_max_class', mx.max()),
                            ('min_max_class', mx.min())])

        if target is not None:
            y_hat = T.argmax(state, axis=1)
            y = T.argmax(target, axis=1)
            misclass = T.neq(y, y_hat).mean()
            misclass = T.cast(misclass, config.floatX)
            rval['misclass'] = misclass
            rval['nll'] = self.cost(Y_hat=state, Y=target)
            rval['ppl'] = 2 ** (rval['nll'] / T.log(2))

        return rval


class ProjectionLayer(Layer):
    """
    This layer can be used to project discrete labels into a continous space
    as done in e.g. language models. It takes labels as an input (IndexSpace)
    and maps them to their continous embeddings and concatenates them.

    Parameters
        ----------
    dim : int
        The dimension of the embeddings. Note that this means that the
        output dimension is (dim * number of input labels)
    layer_name : string
        Layer name
    irange : numeric
       The range of the uniform distribution used to initialize the
       embeddings. Can't be used with istdev.
    istdev : numeric
        The standard deviation of the normal distribution used to
        initialize the embeddings. Can't be used with irange.
    """
    def __init__(self, dim, layer_name, irange=None, istdev=None):
        """
        Initializes a projection layer.
        """
        super(ProjectionLayer, self).__init__()
        self.dim = dim
        self.layer_name = layer_name
        if irange is None and istdev is None:
            raise ValueError("ProjectionLayer needs either irange or"
                             "istdev in order to intitalize the projections.")
        elif irange is not None and istdev is not None:
            raise ValueError("ProjectionLayer was passed both irange and "
                             "istdev but needs only one")
        else:
            self._irange = irange
            self._istdev = istdev

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):

        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if isinstance(space, IndexSpace):
            self.input_dim = space.dim
            self.input_space = space
        else:
            raise ValueError("ProjectionLayer needs an IndexSpace as input")
        self.output_space = VectorSpace(self.dim * self.input_dim)
        rng = self.mlp.rng
        if self._irange is not None:
            W = rng.uniform(-self._irange,
                            self._irange,
                            (space.max_labels, self.dim))
        else:
            W = rng.randn(space.max_labels, self.dim) * self._istdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W, = self.transformer.get_params()
        assert W.name is not None

    @wraps(Layer.fprop)
    def fprop(self, state_below,targets=None):
        z = self.transformer.project(state_below)
        return z

    @wraps(Layer.get_params)
    def get_params(self):
        W, = self.transformer.get_params()
        assert W.name is not None
        params = [W]
        return params

class Tanh(mlp.Tanh):
    """
    A layer that performs an affine transformation of its (vectorial)
    input followed by a hyperbolic tangent elementwise nonlinearity.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to pass through to `Linear` class constructor.
    """
    def fprop(self, state_below,targets=None):

        p = self._linear_part(state_below)
        p = T.tanh(p)
        return p

class MLP(mlp.MLP):
    
    def fprop(self, state_below, targets,return_all=False):
        if not hasattr(self, "input_space"):
            raise AttributeError("Input space has not been provided.")

        rval = self.layers[0].fprop(state_below,targets)
        rlist = [rval]
        for layer in self.layers[1:]:
            rval = layer.fprop(rval,targets)
            rlist.append(rval)

        if return_all:
            return rlist
        return rval

    def cost_from_X(self, data):
        """
        Computes self.cost, but takes data=(X, Y) rather than Y_hat as an
        argument.

        This is just a wrapper around self.cost that computes Y_hat by
        calling Y_hat = self.fprop(X)

        Parameters
        ----------
        data : WRITEME
        """
        self.cost_from_X_data_specs()[0].validate(data)
        X, Y = data
        Y_hat = self.fprop(X,Y)
        return self.cost(Y, Y_hat)

    def get_layer_monitoring_channels(self, state_below=None, state=None, targets=None):

        rval = OrderedDict()
        if state_below is not None:
            state = state_below

            for layer in self.layers:
                # We don't go through all the inner layers recursively
                print type(layer)
                state = layer.fprop(state,targets)
                args = [None, state]
                if layer is self.layers[-1] and targets is not None:
                    args.append(targets)
                ch = layer.get_layer_monitoring_channels(*args)
                if not isinstance(ch, OrderedDict):
                    raise TypeError(str((type(ch), layer.layer_name)))
                for key in ch:
                    value = ch[key]
                    doc = get_monitor_doc(value)
                    if doc is None:
                        doc = str(type(layer)) + \
                            ".get_monitoring_channels_from_state did" + \
                            " not provide any further documentation for" + \
                            " this channel."
                    doc = 'This channel came from a layer called "' + \
                            layer.layer_name + '" of an MLP.\n' + doc
                    value.__doc__ = doc
                    rval[layer.layer_name+'_'+key] = value


        elif state is not None:

            for layer in self.layers:
                if layer is self.layers[-1]:
                    args = [None, state]
                    if targets is not None:
                        args.append(targets)
                    ch = layer.get_layer_monitoring_channels(*args)
                else:
                    ch = layer.get_layer_monitoring_channels()
                for key in ch:
                    value = ch[key]
                    doc = get_monitor_doc(value)
                    if doc is None:
                        doc = str(type(layer)) + \
                            ".get_monitoring_channels did" + \
                            " not provide any further documentation for" + \
                            " this channel."
                    doc = 'This channel came from a layer called "' + \
                            layer.layer_name + '" of an MLP.\n' + doc
                    value.__doc__ = doc
                    rval[layer.layer_name+'_'+key] = value

        else:
            for layer in self.layers:
                ch = layer.get_layer_monitoring_channels()
                if not isinstance(ch, OrderedDict):
                    raise TypeError(str((type(ch), layer.layer_name)))
                for key in ch:
                    value = ch[key]
                    doc = get_monitor_doc(value)
                    if doc is None:
                        doc = str(type(layer)) + \
                            ".get_monitoring_channels_from_state did" + \
                            " not provide any further documentation for" + \
                            " this channel."
                    doc = 'This channel came from a layer called "' + \
                            layer.layer_name + '" of an MLP.\n' + doc
                    value.__doc__ = doc
                    rval[layer.layer_name+'_'+key] = value

        return rval


class ClassBasedOutput(Softmax):
    # TODO cleanup target, class name mess, it's confusing
    def __init__(self, n_clusters = None, classclusterpath= None, clusters_scope = None, **kwargs):
        super(ClassBasedOutput, self).__init__(**kwargs)
        self.n_clusters = n_clusters

        del self.b
        self.b_class = sharedX(np.zeros((self.n_clusters, self.n_classes)), name = 'softmax_b_class')
        self.b_cluster = sharedX( np.zeros((self.n_clusters)), name = 'softmax_b_clusters')
        
        npz_clust = serial.load("${PYLEARN2_DATA_PATH}/PennTreebankCorpus/" + classclusterpath)        
        array_clusters = npz_clust['wordwithclusters']
       
        #z = array_clusters[np.in1d(array_clusters[:,0], self._data[:,-1:]), 1]
        #npz_data = serial.load("/u/huilgolr/data/PennTreebank/processed.npz")
        #print npz_data['word_clusters'].shape
        #self.classclusters=sharedX(npz_data['word_clusters'][:,1],'classclusters')
        #self.cluster_targets = np.random.randint(0,n_clusters,size=(self.n_classes))
        #cluster_targets is a nx1 array which tells which cluster the word

        keys = range(n_clusters)
        self.clusters_scope = dict(zip(keys, np.bincount(array_clusters.astype(int))))
        #self._group_dot = _group_dot
        self.array_clusters = sharedX(array_clusters)
        
    def set_input_space(self, space):
        self.input_space = space
        if not isinstance(space, Space):
            raise TypeError("Expected Space, got "+
                    str(space)+" of type "+str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, VectorSpace)
	
        if self.no_affine:
            desired_dim = self.n_classes
            assert self.input_dim == desired_dim
        else:
            desired_dim = self.input_dim
        self.desired_space = VectorSpace(desired_dim)

        if not self.needs_reformat:
            assert self.desired_space == self.input_space

        rng = self.mlp.rng

        if self.no_affine:
            self._params = []
        else:
            if self.irange is not None:
                assert self.istdev is None
                assert self.sparse_init is None
                W_cluster = rng.uniform(-self.irange,self.irange, (self.input_dim, self.n_clusters))
                W_class = rng.uniform(-self.irange,self.irange, (self.n_clusters, self.input_dim, self.n_classes))
            elif self.istdev is not None:
                assert self.sparse_init is None
                W_cluster = rng.randn(self.input_dim, self.n_clusters) * self.istdev
                W_class = rng.randn(self.n_clusters, self.input_dim, self.n_classes) * self.istdev
            else:
                raise NotImplementedError()

            # set the extra dummy weights to 0
            for key in self.clusters_scope.keys():
		#print key
                #should probably be reverse
                W_class[int(key), :, :self.clusters_scope[key]] = 0.

            self.W_class = sharedX(W_class,  'softmax_W_class' )
            self.W_cluster = sharedX(W_cluster,  'softmax_W_cluster' )

            self._params = [self.b_class, self.W_class, self.b_cluster, self.W_cluster]

    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=NotImplementedError):

        if self.no_affine:
            return OrderedDict()

        W_class = self.W_class
        W_cluster = self.W_cluster

        assert W_class.ndim == 3
        assert W_cluster.ndim == 2

        sq_W = T.sqr(W_cluster)
        sq_W_class = T.sqr(W_class)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        row_norms_class = T.sqrt(sq_W_class.sum(axis=1))
        col_norms_class = T.sqrt(sq_W_class.sum(axis=0))

        rval = OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ('class_row_norms_min'  , row_norms_class.min()),
                            ('class_row_norms_mean' , row_norms_class.mean()),
                            ('class_row_norms_max'  , row_norms_class.max()),
                            ('class_col_norms_min'  , col_norms_class.min()),
                            ('class_col_norms_mean' , col_norms_class.mean()),
                            ('class_col_norms_max'  , col_norms_class.max()),
                            ])


        if (state_below is not None) or (state is not None):
            if state is None:

                for value in get_debug_values(state_below):
                    print 'value is'+ value
                state=self.fprop (state_below,targets)
            #print state
            probclass, probcluster = state
            mx = probclass.max(axis=1)
            rval.update(OrderedDict([('mean_max_class',mx.mean()),
                                     ('max_max_class' , mx.max()),
                                     ('min_max_class' , mx.min())
                                    ]))
            if targets is not None:
                rval['nll'] = self.cost(Y=targets,Y_hat=(probclass,probcluster))
                rval['perplexity'] = 10 ** (rval['nll']/np.log(10).astype('float32'))
                rval['entropy'] = rval['nll']/np.log(2).astype('float32')
        return rval
        
    def cost(self, Y, Y_hat):
        """
        Y must be one-hot binary. Y_hat is a softmax estimate.
        of Y. Returns negative log probability of Y under the Y_hat
        distribution.
        """
        y_probclass, y_probcluster = Y_hat
        #separated
        #have to change y as argmax
        #also make cls a shared variable and use that
        #Y,
        
        #CLS = self.classclusters[Y]
        #Y = self._group_dot.fprop(Y, Y_hat)
        
        CLS = self.array_clusters[T.cast(T.argmax(Y),'int32')]

        assert hasattr(y_probclass, 'owner')
        owner = y_probclass.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
          assert len(owner.inputs) == 1
          y_probclass, = owner.inputs
          owner = y_probclass.owner
          op = owner.op
        assert isinstance(op, T.nnet.Softmax)

        z_class ,= owner.inputs
        assert z_class.ndim == 2

        assert hasattr(y_probcluster, 'owner')
        owner = y_probcluster.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            y_probcluster, = owner.inputs
            owner = y_probcluster.owner
            op = owner.op
        assert isinstance(op, T.nnet.Softmax)
        z_cluster ,= owner.inputs
        assert z_cluster.ndim == 2

        z_class = z_class - z_class.max(axis=1).dimshuffle(0, 'x')
        log_prob = z_class - T.log(T.exp(z_class).sum(axis=1).dimshuffle(0, 'x'))
        # we use sum and not mean because this is really one variable per row
        # Y = OneHotFormatter(self.n_classes).theano_expr(
        #                         T.addbroadcast(Y,0,1).dimshuffle(0).astype('uint32'))
        log_prob_of = (Y * log_prob).sum(axis=1)
        assert log_prob_of.ndim == 1

        # cls
        z_cluster = z_cluster - z_cluster.max(axis=1).dimshuffle(0, 'x')
        log_prob_cls = z_cluster - T.log(T.exp(z_cluster).sum(axis=1).dimshuffle(0, 'x'))

        # CLS = OneHotFormatter(self.n_clusters).theano_expr(
        #                         T.addbroadcast(CLS, 1).dimshuffle(0).astype('uint32'))
        log_prob_of_cls = (CLS * log_prob_cls).sum(axis=1)
        assert log_prob_of_cls.ndim == 1

        # p(w|history) = p(c|s) * p(w|c,s)
        log_prob_of = log_prob_of + log_prob_of_cls
        rval = log_prob_of.mean()        
        return - rval

    def fprop(self, state_below,targets):
        #change model to add new variable which sends which indices of the data are here
        self.input_space.validate(state_below)        
        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)
        for value in get_debug_values(state_below):
            if self.mlp.batch_size is not None and value.shape[0] != self.mlp.batch_size:
                raise ValueError("state_below should have batch size "+str(self.dbm.batch_size)+" but has "+str(value.shape[0]))
        self.desired_space.validate(state_below)
        
        assert state_below.ndim == 2
        if not hasattr(self, 'no_affine'):
            self.no_affine = False
        if self.no_affine:
            raise NotImplementedError()

        assert self.W_class.ndim == 3
        assert self.W_cluster.ndim == 2

        #we get the cluster by doing hW_cluster + b_cluster
        probcluster = T.dot(state_below, self.W_cluster) + self.b_cluster
        probcluster = T.nnet.softmax(probcluster)
        
        #need the predicted clusters for this batch
        if targets is not None:
            batch_clusters = self.array_clusters[T.cast(T.argmax(targets).flatten(),'int32')]
            Z = T.nnet.GroupDot(self.n_clusters, gpu='gpu' in theano.config.device)(state_below,
                                                        self.W_class,
                                                        self.b_class,
                                                        T.cast(batch_clusters,'int32'))
        probclass = T.nnet.softmax(Z)
        
        for value in get_debug_values(probclass):
             if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size
        return probclass, probcluster

    def get_weights_format(self):
        return ('v', 'h', 'h_c')

    def get_biases(self):
        return self.b_class.get_value(), self.b_cluster.get_value()

    def get_weights(self):
        if not isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()
        return self.W_cluster.get_value(), self.W_class.get_value()
