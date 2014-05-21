class FactorizedSoftmax(Softmax):
    # TODO cleanup target, class name mess, it's confusing
    def __init__(self, n_clusters = None, clusters_scope = None, **kwargs):
        super(FactorizedSoftmax, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.clusters_scope = clusters_scope
        del self.b
        self.b_class = sharedX(np.zeros((self.n_clusters, self.n_classes)), name = 'softmax_b_class')
        self.b_cluster = sharedX( np.zeros((self.n_clusters)), name = 'softmax_b_clusters')
        self.output_space = VectorSpace(1)

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
                W_class[int(key), :, :self.clusters_scope[key]] = 0.

            self.W_class = sharedX(W_class,  'softmax_W_class' )
            self.W_cluster = sharedX(W_cluster,  'softmax_W_cluster' )

            self._params = [self.b_class, self.W_class, self.b_cluster, self.W_cluster]

    def get_monitoring_channels(self):

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

        return OrderedDict([
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
                            ('clas_col_norms_max'  , col_norms_class.max()),

                            ])

    def get_weights(self):
        if not isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()

        return self.W_class.get_value(), self. W_cluster.get_value()

    def cost(self, Y, Y_hat):
        """
        Y must be one-hot binary. Y_hat is a softmax estimate.
        of Y. Returns negative log probability of Y under the Y_hat
        distribution.

        Parameters
        ----------
        Y : WRITEME
        Y_hat : WRITEME

        Returns
        -------
        WRITEME
        """
        y_hat, y_cls = Y_hat
        Y, CLS = Y
        assert hasattr(y_hat, 'owner')
        owner = y_hat.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            y_hat, = owner.inputs
            owner = y_hat.owner
            op = owner.op
        assert isinstance(op, T.nnet.Softmax)
        z ,= owner.inputs
        assert z.ndim == 2

        assert hasattr(y_cls, 'owner')
        owner = y_cls.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            y_cls, = owner.inputs
            owner = y_cls.owner
            op = owner.op
        assert isinstance(op, T.nnet.Softmax)
        z_cls ,= owner.inputs
        assert z_cls.ndim == 2

        # Y
        print z
        z = z - z.max(axis=1).dimshuffle(0, 'x')
        print z
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        print log_prob
        # we use sum and not mean because this is really one variable per row
        Y = OneHotFormatter(self.n_classes).theano_expr(
                                T.addbroadcast(Y, 1).dimshuffle(0).astype('uint32'))
        log_prob_of = (Y * log_prob).sum(axis=1)
        assert log_prob_of.ndim == 1

        # cls
        z_cls = z_cls - z_cls.max(axis=1).dimshuffle(0, 'x')
        log_prob_cls = z_cls - T.log(T.exp(z_cls).sum(axis=1).dimshuffle(0, 'x'))

        CLS = OneHotFormatter(self.n_clusters).theano_expr(
                                T.addbroadcast(CLS, 1).dimshuffle(0).astype('uint32'))
        log_prob_of_cls = (CLS * log_prob_cls).sum(axis=1)
        assert log_prob_of_cls.ndim == 1

        # p(w|history) = p(c|s) * p(w|c,s)
        log_prob_of = log_prob_of + log_prob_of_cls
        rval = log_prob_of.mean()

        return - rval

    def get_monitoring_channels_from_state(self, state, target=None, cluster_tragets = None):
        """
        .. todo::

            WRITEME
        """

        state, cls = state
        mx = state.max(axis=1)

        rval =  OrderedDict([
                ('mean_max_class' , mx.mean()),
                ('max_max_class' , mx.max()),
                ('min_max_class' , mx.min())
        ])

        if target is not None:
            rval['nll'] = self.cost(Y_hat=(state, cls), Y=(target, cluster_tragets))
            rval['perplexity'] = 10 ** (rval['nll'] / np.log(10).astype('float32'))
            rval['entropy'] = rval['nll'] / np.log(2).astype('float32')

        return rval

    def fprop(self, state_below, cluster_tragetss):
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

        cls = T.dot(state_below, self.W_cluster) + self.b_cluster
        cls = T.nnet.softmax(cls)

        Z = GroupDot(self.n_clusters,
                gpu='gpu' in theano.config.device)(state_below,
                                                    self.W_class,
                                                    self.b_class,
                                        cluster_tragetss.flatten().astype('uint32'))
        rval = T.nnet.softmax(Z)

        for value in get_debug_values(rval):
             if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size

        return rval, cls

    def censor_updates(self, updates):
        #if self.no_affine:
            #return
        #if self.max_row_norm is not None:
            #W = self.W
            #if W in updates:
                #updated_W = updates[W]
                #row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                #desired_norms = T.clip(row_norms, 0, self.max_row_norm)
                #updates[W] = updated_W * (desired_norms / (1e-7 + row_norms)).dimshuffle(0, 'x')
        #if self.max_col_norm is not None:
            #assert self.max_row_norm is None
            #W = self.W
            #if W in updates:
                #updated_W = updates[W]
                #col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                #desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                #updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))
        return

    def get_weights_format(self):
        return ('v', 'h', 'h_c')

    def get_biases(self):
        return self.b_class.get_value(), self.b_cluster.get_value()

    def get_weights(self):
        if not isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()

        return self.W_cluster.get_value(), self.W_class.get_value()

