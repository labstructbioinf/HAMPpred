import os

from hamp_pred.src.predictor_config import PredictionConfig


def get_config(predictor):
    prev_config = predictor.config
    last_config = PredictionConfig.from_pickle(os.path.join(prev_config.model_config['data_dir'], 'config.p'))
    last_config = last_config.merge_with(prev_config, favour_other=False)
    last_config.model_config['data_dir'] = prev_config.model_config['data_dir']
    return last_config
