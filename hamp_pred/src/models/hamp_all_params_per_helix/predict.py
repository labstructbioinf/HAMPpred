import numpy as np

from hamp_pred.src.models.common.models import BaseLinearWrapper
from .config import get_config


def run(sequences, config=None):
    main_config = get_config(config.get('predictor')).dump()
    main_config['is_test'] = config.get('is_test')
    config = main_config
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    n_chains, features = config.get('n_chains', 2), 20
    operator.n_chains = n_chains
    to_pred = operator.get_for_prediction(sequences)
    model = model or BaseLinearWrapper(config=config)
    inp_shape = to_pred.shape[1], to_pred.shape[-1]
    md = model.build(inp_shape, features).compile(optimizer="Adam", loss="mse", metrics=["mae"])
    prediction = md.predict(to_pred)
    features = ["n_crick_mut", "n_shift", "n_radius", "n_pitch", "n_P",
                "c_crick_mut", "c_shift", "c_radius", "c_pitch", "c_P"]
    result = operator.get_from_prediction(prediction, n_features=10, shrink_factor=1,
                                          result_col='predicted_params',
                                          feature_names=features), md, to_pred
    return result
