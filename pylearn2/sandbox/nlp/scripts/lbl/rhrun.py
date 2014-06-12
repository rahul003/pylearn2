import os
import theano
import pylearn2
from pylearn2.termination_criteria import EpochCounter
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.config import yaml_parse

theano.config.exception_verbosity = 'high'

skip_if_no_data()
#/u/huilgolr/
with open('/home/rh/git/pylearn2/pylearn2/sandbox/nlp/scripts/lbl/lbl.yaml', 'r') as f:
    train_3 = f.read()
#train_3 = train_3 % (hyper_params)
train_3 = yaml_parse.load(train_3)
train_3.main_loop()
