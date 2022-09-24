import logging
import pathlib

import numpy as np

from ma_predictor.src.input_prep.prepare_sequence import MultiChainOperator
from ma_predictor.src.models.common.models import BaseLinearWrapper


def run(data, config=None):
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    n_chains, features = config.get('n_chains', 2), config.get('n_features', 2)
    operator.n_chains = n_chains
    operator.solo_helix=True
    helix_data = operator.get_for_train(data)
    train, val, test = helix_data
    inp_shape, out_shape = train[0][0].shape[1:], train[1][0].shape[1:]
    model = model or BaseLinearWrapper(config=config)
    model.build_many_inp_out(inp_shape, 1).compile(optimizer="Adam", loss="mse", metrics=["mae"]). \
        train(train[0], train[1], val[0], val[1])
    results = model.test(test[0], test[1])
    return results

