import pathlib

from ma_predictor.src.input_prep.prepare_sequence import MultiChainOperator
from ma_predictor.src.models.common.models import BaseLinearWrapper
import pandas as pd

def run(data, config=None):
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    n_chains, features = config.get('n_chains', 2), 1
    model.config['activation'] = 'sigmoid'
    operator.n_chains = 1
    operator.chain_names = ('N',)
    seqs = list(data.seq.values)
    vals = [ [[ [int(p)] for p in x]] for x in data['labels'].values]
    operator.y_encoder = None
    train, valid, test = operator.get_for_train(seqs, vals,
                                                test_ids=config.get('test_ids'))
    X, y = train
    val_x, val_y = valid
    model = model or BaseLinearWrapper(config=config)
    inp_shape = X.shape[1], X.shape[-1]
    model.build(inp_shape, 1).compile(optimizer="Adam", loss="binary_crossentropy", metrics=["binary_crossentropy"]). \
        train(X, y, val_x, val_y)
    results = model.test(test[0], test[1])
    return results


