from hamp_pred.src.input_prep.encode import MultiEncoder, OneHotEncoderSeq, RadianEncoder, RadiousPhobosEncoder
from hamp_pred.src.input_prep.prepare_sequence import MultiChainOperator, SeqWindow
from hamp_pred.src.models.common.models import BaseConvolutionalWrapper
from hamp_pred.src.predictor_config import PredictionConfig


def get_config(predictor):
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
    last_config = PredictionConfig(BaseConvolutionalWrapper, operator, model_conf)
    prev_config = predictor.config
    last_config = last_config.merge_with(prev_config, favour_other=False)
    last_config.model_config['data_dir'] = prev_config.model_config['data_dir']
    return last_config
