import importlib
import os

import numpy as np
import pandas as pd
import seaborn

from hamp_pred.src.input_prep.encode import MultiEncoder, OneHotEncoderSeq, RadianEncoder, RadiousPhobosEncoder
from hamp_pred.src.input_prep.prepare_sequence import MultiChainOperator, SeqWindow
from hamp_pred.src.models.common.models import BaseConvolutionalWrapper
from hamp_pred.src.output_analysis.common_processors import FastaProcessor, MsaProcessor, SamccTestProcessor
from hamp_pred.src.output_analysis.feature_importance import ImportanceDescriber, ModelMetrics
from hamp_pred.src.predictor_config import PredictionConfig


class Predictor:

    def __init__(self, model, version=None,
                 model_data_dir=None,
                 config=None, infra=None, processors=None):
        pos = os.path.dirname(__file__)
        self.models_dir = os.path.join(pos, 'models')
        self.model = model
        self.version = version
        self.model_dir = os.path.join(os.path.basename(self.models_dir), self.model)
        self.model_data_dir = model_data_dir or self.default_model_dir()
        self.config = config
        self.processors = processors or {'msa': MsaProcessor,
                                         'fasta': FastaProcessor,
                                         'samcc_test': SamccTestProcessor,
                                         'mutator': None,
                                         'importance_describer': ImportanceDescriber,
                                         'metrics': ModelMetrics}
        if not os.path.exists(self.models_dir):
            raise ValueError(f"Model path {self.model_dir}. Available:"
                             f"{set(os.listdir(self.models_dir)).difference({'common'})}")
        self.infra = infra

    def default_model_dir(self):
        pos = os.path.dirname(__file__)
        if self.version:
            return os.path.abspath(os.path.join(pos, '../../', f'data/output/weights/{self.model}/{self.version}'))
        return os.path.abspath(os.path.join(pos, '../../', f'data/output/weights/{self.model}'))

    def predict(self, data, with_model=False, **kwargs):
        base = self.model_dir.replace(os.sep, '.')
        predict = importlib.import_module(f"hamp_pred.src.{base}.predict")
        self._prepare_config_for_predict()
        last_config = PredictionConfig.from_pickle(os.path.join(self.model_data_dir, 'config.p'))
        if self.config:
            last_config = last_config.merge_with(self.config, favour_other=True)
        conf = last_config.dump()
        conf |= kwargs
        for kw in kwargs:
            if hasattr(conf['operator'], kw):
                setattr(conf['operator'], kw, kwargs[kw])
        result = predict.run(data, conf)
        if with_model:
            return result
        return result[0]

    def _prepare_config_for_train(self, ids, val_ids):
        self.config.set_task(self.model, self.version)
        self.config.model_config['data_dir'] = self.model_data_dir
        self.config.set_val_ids(val_ids)
        self.config.set_ids(ids)
        self.config.set_test_ids()
        self.config.operator.parallel = True

    def _prepare_config_for_predict(self):
        self.config = self.config or PredictionConfig(None, None)
        self.config.model_config['data_dir'] = self.model_data_dir

    def train(self, data, ids=None, val_ids=None):
        base = self.model_dir.replace(os.sep, '.')
        if self.config is None:
            raise AttributeError("Provide config for train.")
        self._prepare_config_for_train(ids, val_ids)
        train = importlib.import_module(f"hamp_pred.src.{base}.train")
        result = train.run(data, self.config.dump())
        self.config.as_pickle(os.path.join(self.model_data_dir, 'config.p'))
        return result

    def process_data(self, data, *args, kind='msa', **kwargs):
        processor = self.processors.get(kind)(*args, **kwargs, model=self)
        if hasattr(processor, 'prepare_pred'):
            to_pred = processor.prepare_pred(data)
        else:
            to_pred = data
        conf = self.config
        if hasattr(processor, 'adjust_config'):
            self.config = processor.adjust_config(self.config)
        results = self.predict(to_pred)
        self.config = conf
        if hasattr(processor, 'prepare_out'):
            return processor.prepare_out(results)
        return results

    def train_processed(self, *args, kind='msa', **kwargs):
        processor = self.processors.get(kind)(*args, **kwargs)
        to_train = processor.prepare_train()
        results = self.train(to_train)
        return results

    def md(self, data, **kwargs):
        base = self.model_dir.replace(os.sep, '.')
        predict = importlib.import_module(f"hamp_pred.src.{base}.predict")
        conf = self.config.dump()
        conf |= kwargs
        for kw in kwargs:
            if hasattr(conf['operator'], kw):
                setattr(conf['operator'], kw, kwargs[kw])
        return predict.model(data, conf)[0]

data = pd.read_pickle('/Users/awinski/PycharmProjects/HAMPred/data/input/full_alpha_data.p').head(2)
operator = MultiChainOperator(MultiEncoder([RadiousPhobosEncoder(), OneHotEncoderSeq()]), SeqWindow(140, 140), RadianEncoder(100),  SeqWindow(140, 140, null_char=[[0]]),
                                      parallel=True, n_chains=1, chain_names=('seq', ))
model_conf = model_config = {
    'activation': 'linear',
    'norm': True,
    'n_layers': 1,
    'kernel_sizes': (3, 5, 7),
    'lstm': 2,
    'dense': 1,
    'reshape_out': False,
    'epochs': 600
}
conf = PredictionConfig(BaseConvolutionalWrapper, operator, model_conf)
# seq = "LKELVQGVQRIIGELITSFNLM"
# dd = pd.DataFrame({'train_seq': [seq], 'n_crick_mut': n, 'c_crick_mut': c} )
pred = Predictor('hamp_crick_single_sequence', config=conf)
# pred.train(data)
w = pred.predict([data.iloc[0]['sequence']])
pass