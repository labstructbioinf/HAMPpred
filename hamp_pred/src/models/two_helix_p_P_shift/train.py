import pathlib

from ma_predictor.src.input_prep.prepare_sequence import MultiChainOperator
from ma_predictor.src.models.common.models import BaseLinearWrapper


def run(data, config=None):
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    n_chains, features = config.get('n_chains', 2), config.get('n_features', 2)
    operator.n_chains = n_chains
    train, valid, test = operator.get_for_train(data,
                                                test_ids=config.get('test_ids'),
                                                source_config={'params': ('p', 'P', 'shift')})
    X, y = train
    val_x, val_y = valid
    model = model or BaseLinearWrapper(config=config)
    inp_shape = X.shape[1], X.shape[-1]
    model.build(inp_shape, 6).compile(optimizer="Adam", loss="mse", metrics=["mae"]). \
        train(X, y, val_x, val_y)
    results = model.test(test[0], test[1])
    return results
