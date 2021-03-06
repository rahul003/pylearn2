from theano import tensor as T
import theano
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin, NullDataSpecsMixin
from pylearn2.utils import safe_izip


class Default(DefaultDataSpecsMixin, Cost):
    """
    The default Cost to use with an MLP.

    It simply calls the MLP's cost_from_X method.
    """

    supervised = True

    def expr(self, model, data, **kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        return model.cost_from_X(data)

class Dropout(DefaultDataSpecsMixin, Cost):
    """
    Implements the dropout training technique described in
    "Improving neural networks by preventing co-adaptation of feature
    detectors"
    Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever,
    Ruslan R. Salakhutdinov
    arXiv 2012

    This paper suggests including each unit with probability p during training,
    then multiplying the outgoing weights by p at the end of training.
    We instead include each unit with probability p and divide its
    state by p during training. Note that this means the initial weights should
    be multiplied by p relative to Hinton's.
    The SGD learning rate on the weights should also be scaled by p^2 (use
    W_lr_scale rather than adjusting the global learning rate, because the
    learning rate on the biases should not be adjusted).

    During training, each input to each layer is randomly included or excluded
    for each example. The probability of inclusion is independent for each
    input and each example. Each layer uses "default_input_include_prob"
    unless that layer's name appears as a key in input_include_probs, in which
    case the input inclusion probability is given by the corresponding value.

    Each feature is also multiplied by a scale factor. The scale factor for
    each layer's input scale is determined by the same scheme as the input
    probabilities.

    Parameters
    ----------
    default_input_include_prob : float
        The probability of including a layer's input, unless that layer appears
        in `input_include_probs`
    input_include_probs : dict
        A dictionary mapping string layer names to float include probability
        values. Overrides `default_input_include_prob` for individual layers.
    default_input_scale : float
        During training, each layer's input is multiplied by this amount to
        compensate for fewer of the input units being present. Can be
        overridden by `input_scales`.
    input_scales : dict
        A dictionary mapping string layer names to float values to scale that
        layer's input by. Overrides `default_input_scale` for individual
        layers.
    per_example : bool
        If True, chooses separate units to drop for each example. If False,
        applies the same dropout mask to the entire minibatch.
    """

    supervised = True

    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
            default_input_scale=2., input_scales=None, per_example=True):

        print type(default_input_scale)

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):

        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        #thisis q_h
        Y_hat = model.dropout_fprop(
            model.projector_context.project(X),
            default_input_include_prob=self.default_input_include_prob,
            input_include_probs=self.input_include_probs,
            default_input_scale=self.default_input_scale,
            input_scales=self.input_scales,
            per_example=self.per_example
        )

        #cost projects y to get q_w and then gives cost of q_w vs _wh
        return model.cost(Y, Y_hat)
