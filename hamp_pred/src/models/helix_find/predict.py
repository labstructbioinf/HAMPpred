import numpy as np

from hamp_pred.src.input_prep.encode import OneHotEncoderSeq, RadianEncoder
from hamp_pred.src.input_prep.prepare_sequence import MultiChainOperator, SeqWindow
from hamp_pred.src.models.common.models import BaseLinearWrapper


def run(sequences, config=None):
    mod, to_pred, operator = model(sequences, config)
    prediction = mod.predict(to_pred)
    return operator.get_from_prediction(prediction, n_features=1, shrink_factor=1)


def model(sequences, config=None):
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    model.config['activation'] = 'sigmoid'
    n_chains, features = config.get('n_chains', 2), 1
    operator.n_chains = 1
    operator.chain_names = ('N',)
    operator.y_encoder = None
    to_pred = operator.get_for_prediction(sequences)
    model = model or BaseLinearWrapper(config=config)
    inp_shape = to_pred.shape[1], to_pred.shape[-1]
    model = model.build(inp_shape, features).compile(optimizer="Adam", loss="binary_crossentropy", metrics=["mae"])
    return model, to_pred, operator
