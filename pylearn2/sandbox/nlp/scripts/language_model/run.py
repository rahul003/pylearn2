import os
import theano
import pylearn2
from pylearn2.termination_criteria import EpochCounter
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.config import yaml_parse

CUDA_LAUNCH_BLOCKING=1
theano.config.exception_verbosity = 'high'
skip_if_no_data()
with open('/u/huilgolr/pylearn2/pylearn2/sandbox/nlp/scripts/language_model/class_soft.yaml', 'r') as f:
    train_3 = f.read()
#train_3 = train_3 % (hyper_params)
train_3 = yaml_parse.load(train_3)
train_3.main_loop()
