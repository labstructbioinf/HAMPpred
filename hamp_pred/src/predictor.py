import importlib
import os

from hamp_pred.src.output_analysis.common_processors import FastaProcessor, MsaProcessor, SamccTestProcessor
from hamp_pred.src.output_analysis.feature_importance import ImportanceDescriber, ModelMetrics
from hamp_pred.src.predictor_config import PredictionConfig


class Predictor:
    pos = os.path.dirname(__file__)
    models_dir = os.path.join(pos, 'models')
    def __init__(self, model, version=None,
                 model_data_dir=None,
                 config=None, infra=None, processors=None):
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
        if self.config:
            self.config.predictor = self

    def default_model_dir(self):
        pos = os.path.dirname(__file__)
        if self.version:
            return os.path.abspath(os.path.join(pos, '../../', f'data/output/weights/{self.model}/{self.version}'))
        return os.path.abspath(os.path.join(pos, '../../', f'data/output/weights/{self.model}'))

    def predict(self, data, with_model=False, only_encode=False, **kwargs):
        base = self.model_dir.replace(os.sep, '.')
        predict = importlib.import_module(f"hamp_pred.src.{base}.predict")
        self._prepare_config_for_predict()
        conf_path = os.path.join(self.model_data_dir, 'config.p')
        if os.path.exists(conf_path):
            last_config = PredictionConfig.from_pickle(os.path.join(self.model_data_dir, 'config.p'))
        else:
            last_config = self.config
        if self.config:
            last_config = last_config.merge_with(self.config, favour_other=True)
        conf = last_config.dump()
        conf |= kwargs
        for kw in kwargs:
            if hasattr(conf['operator'], kw):
                setattr(conf['operator'], kw, kwargs[kw])
        if kwargs.get('is_test'):
            conf['is_test'] = True
        for param in ['tester', 'pred_adjuster']:
            conf[param] = kwargs.get(param)
        if only_encode:
            return conf['operator'].get_for_prediction(data)
        result = predict.run(data, conf)
        if kwargs.get('is_test'):
            conf['is_test'] = False
            return result
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

    @classmethod
    def get_models_info(cls):
        return [{"name": mod } for mod in set(os.listdir(cls.models_dir)).difference({'common'}) if not mod.startswith('__')]