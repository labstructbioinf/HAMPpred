import numpy as np

from ma_predictor.src.input_prep.encode import OneHotEncoderSeq, RadianEncoder
from ma_predictor.src.input_prep.prepare_sequence import MultiChainOperator, SeqWindow
from ma_predictor.src.models.common.models import  BaseLinearWrapper


def run(sequences, config=None):
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    n_chains, features = config.get('n_chains', 2), config.get('n_features', 2)
    operator.n_chains = n_chains
    operator.solo_helix = True
    to_preds = operator.get_for_prediction(sequences)
    sh = to_preds[0].shape[1:]
    predictions = model.build_many_inp_out(sh, features//n_chains).compile(optimizer="Adam", loss="mse", metrics=["mae"]).predict(to_preds)
    prediction = np.concatenate(predictions, axis=-1)
    return operator.get_from_prediction(prediction, n_features=features//n_chains)