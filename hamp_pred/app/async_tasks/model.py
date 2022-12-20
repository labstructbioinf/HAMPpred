import json

from hamp_pred.app.app import celery
from hamp_pred.src.input_prep.encode import MultiEncoder, OneHotEncoderSeq, RadianEncoder, RadiousPhobosEncoder
from hamp_pred.src.input_prep.prepare_sequence import MultiChainOperator, SeqWindow
from hamp_pred.src.models.common.models import BaseConvolutionalWrapper
from hamp_pred.src.predictor import Predictor
from hamp_pred.src.predictor_config import PredictionConfig
from hamp_pred.utils.numpy_json import NumpyEncoder


@celery.task(bind=True)
def predict_async(celer, model_name, sequences):
    """Async long task method."""
    model = setup(model_name) or Predictor(model_name)
    result = model.predict(sequences)
    result = json.loads(json.dumps(result.to_dict(orient='records'), cls=NumpyEncoder))
    return result


def setup(model_name):
    if model_name == 'hamp_rot' or model_name == 'hamp_crick_ensemble':
        operator = MultiChainOperator(MultiEncoder([RadiousPhobosEncoder(), OneHotEncoderSeq()]), SeqWindow(11, 11),
                                      RadianEncoder(100), SeqWindow(11, 11, null_char=[[0]]),
                                      parallel=True)
        model_conf = {
            'activation': 'tanh',
            'norm': True,
            'n_layers': 1,
            'kernel_sizes': (3, 5, 7),
            'lstm': 2,
            'dense': 1,
            'reshape_out': False,
            'epochs': 100
        }
        conf = PredictionConfig(BaseConvolutionalWrapper, operator, model_conf)
        short_model = Predictor(model_name, config=conf)
        return short_model
    return Predictor(model_name)
