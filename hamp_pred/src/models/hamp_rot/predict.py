import numpy as np

from hamp_pred.src.input_prep.encode import OneHotEncoderSeq, RadianEncoder
from hamp_pred.src.input_prep.prepare_sequence import MultiChainOperator, SeqWindow
from hamp_pred.src.models.common.models import BaseLinearWrapper
from hamp_pred.src.models.hamp_rot.test import Tester


def run(sequences, config=None):
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    n_chains, features = config.get('n_chains', 2), 2
    operator.n_chains = n_chains
    if config.get('is_test', False):
        tester = config.get('tester') or Tester()
        test_data = sequences
        sequences = tester.get_squences(sequences)
    to_pred = operator.get_for_prediction(sequences)
    model = model or BaseLinearWrapper(config=config)
    inp_shape = to_pred.shape[1], to_pred.shape[-1]
    md = model.build(inp_shape, features).compile(optimizer="Adam", loss="mse", metrics=["mae"])
    prediction = md.predict(to_pred)
    prediction = np.concatenate([prediction, prediction], axis=-1)
    result = operator.get_from_prediction(prediction, n_features=2, shrink_factor=2), md, to_pred
    if config.get('is_test', False):
        return tester.get_metrics(test_data, result[0])
    return result