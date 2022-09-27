import numpy as np

from hamp_pred.src.models.common.models import BaseLinearWrapper


def run(sequences, config=None):
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    n_chains, features = operator.n_chains, 2
    to_pred = operator.get_for_prediction(sequences)
    model = model or BaseLinearWrapper(config=config)
    inp_shape = to_pred.shape[1], to_pred.shape[-1]
    md = model.build(inp_shape, features).compile(optimizer="Adam", loss="mse", metrics=["mae"])
    prediction = md.predict(to_pred)
    return operator.get_from_prediction(prediction, n_features=2, shrink_factor=2), md, to_pred
