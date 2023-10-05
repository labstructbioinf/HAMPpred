import numpy as np

from hamp_pred.src.models.common.models import BaseLinearWrapper
from .config import get_config


def run(sequences, config=None):
    main_config = get_config(config.get('predictor')).dump()
    main_config['is_test'] = config.get('is_test')
    config = main_config
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    n_chains, features = config.get('n_chains', 2), 8
    operator.n_chains = n_chains
    to_pred = operator.get_for_prediction(sequences)
    model = model or BaseLinearWrapper(config=config)
    inp_shape = to_pred.shape[1], to_pred.shape[-1]
    md = model.build(inp_shape, features).compile(optimizer="Adam", loss="mse", metrics=["mae"])
    prediction = md.predict(to_pred)
    prediction = np.concatenate([prediction, prediction], axis=-1)
    result = operator.get_from_prediction(prediction, n_features=8, shrink_factor=2,
                                          result_col='predicted_params',
                                          selected_chain_for_results="N_pred",
                                          feature_names=["rot","shift_diff", "radius_diff", "pitch_diff"]), md, to_pred
    return result
