import os

import pylearn2
from pylearn2.termination_criteria import EpochCounter
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.config import yaml_parse


SAVE_PATH = os.path.dirname(os.path.realpath(__file__))

skip_if_no_data()
with open('/u/huilgolr/pylearn2/pylearn2/scripts/tutorials/multilayer_perceptron/mlp_tutorial_part_4.yaml', 'r') as f:
    train_3 = f.read()
hyper_params = {'train_stop': 50,
		'valid_stop': 50050,
		'dim_h0': 5,
		'dim_h1': 10,
		'sparse_init_h1': 2,
		'max_epochs': 1,
		'save_path': SAVE_PATH}
train_3 = train_3 % (hyper_params)
train_3 = yaml_parse.load(train_3)
train_3.main_loop()
cleaunup("mlp_3_best.pkl")
