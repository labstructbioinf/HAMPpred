import numpy as np

from ma_predictor.src.models.two_helix_crick import conf


def run(sequences, config=None):
    model_conf = config.get('model_config')
    model_conf |= conf
    model, operator = config.get('model')(config=model_conf), config.get('operator')
    n_chains, features = config.get('n_chains', 2), 2
    operator.n_chains = n_chains
    to_pred = operator.get_for_prediction(sequences)
    # cc = operator.get_for_prediction(ww)
    # assert np.all(to_pred[0] == cc[0])

    inp_shape = to_pred.shape[1], to_pred.shape[-1]
    model.name = 'base_convolutional_N'
    prediction_N = model.build(inp_shape, features).compile(optimizer="Adam", loss="mse", metrics=["mae"]).predict(
        to_pred)
    model_ref = config.get('model')(config=model_conf)
    model_ref.name = 'base_convolutional_C'
    prediction_C = model_ref.build(inp_shape, features).compile(optimizer="Adam", loss="mse", metrics=["mae"]).predict(
        to_pred)
    prediction = np.concatenate([prediction_N, prediction_C], axis=-1)
    return operator.get_from_prediction(prediction, n_features=(4 // n_chains), shrink_factor=2)
