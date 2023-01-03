import numpy as np

from hamp_pred.src.models.common.models import BaseLinearWrapper
from .config import get_config
from .test import Tester


def run(sequences, config=None):
    main_config = get_config(config.get('predictor')).dump()
    main_config['is_test'] = config.get('is_test')
    config = main_config
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    n_chains, features = config.get('n_chains', 2), 2
    operator.n_chains = n_chains
    if config.get('is_test', False):
        tester = config.get('tester') or Tester(scale=1, out_column='predicted_rotation')
        test_data = sequences
        sequences = tester.get_squences(sequences)
    to_pred = operator.get_for_prediction(sequences)
    model = model or BaseLinearWrapper(config=config)
    inp_shape = to_pred.shape[1], to_pred.shape[-1]
    md = model.build(inp_shape, features).compile(optimizer="Adam", loss="mse", metrics=["mae"])
    prediction = md.predict(to_pred)
    prediction = np.concatenate([prediction, prediction], axis=-1)
    result = operator.get_from_prediction(prediction, n_features=2, shrink_factor=2,
                                          result_col='predicted_rotation'), md, to_pred
    result[0]['predicted_rotation'] = result[0]['N_pred']
    result[0].drop(['N_pred', 'C_pred'], axis='columns')
    if config.get('is_test', False):
        return tester.get_metrics(test_data, result[0])
    return result
