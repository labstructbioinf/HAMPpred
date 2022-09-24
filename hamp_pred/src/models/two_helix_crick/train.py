import numpy as np

from ma_predictor.src.models.common.models import BaseLinearWrapper
from ma_predictor.src.models.two_helix_crick import conf


def run(data, config=None):
    model_conf = config.get('model_config')
    model_conf |= conf
    model, operator = config.get('model')(config=model_conf), config.get('operator')
    n_chains, features = config.get('n_chains', 2), config.get('n_features', 2)
    features = 2
    operator.n_chains = n_chains
    train, valid, test = operator.get_for_train(data,
                                                test_ids=config.get('test_ids'),
                                                source_config={'params': ('crick',)})
    X, y = train
    y_N = y[:,:, 0:2]
    y_C = y[:,:, 2:]

    val_x, val_y = valid
    val_y_N, val_y_C = val_y[:,:, 0:2],val_y[:,:, 2:]
    test_y_N, test_y_C = test[1][:,:, 0:2],test[1][:,:, 2:]
    inp_shape = X.shape[1], X.shape[-1]
    model.name = 'base_convolutional_N'
    model.build(inp_shape, features).compile(optimizer="Adam", loss="mse", metrics=["mae"]). \
        train(X, y_N, val_x, val_y_N)
    model_ref = config.get('model')(config=model_conf)
    model_ref.name = 'base_convolutional_C'
    model_ref.build(inp_shape, features).compile(optimizer="Adam", loss="mse", metrics=["mae"]). \
        train(X, y_C, val_x, val_y_C)
    results = np.concatenate([model.test(test[0], test_y_N), model_ref.test(test[0], test_y_C)], axis=-1)
    return results
