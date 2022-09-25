import importlib
import os

from hamp_pred.src.output_analysis.common_processors import FastaProcessor, MsaProcessor, SamccTestProcessor
from hamp_pred.src.output_analysis.feature_importance import ImportanceDescriber


class Predictor:

    def __init__(self, model, version=None,
                 config=None, infra=None, processors=None):
        pos = os.path.dirname(__file__)
        self.models_dir = os.path.join(pos, 'models')
        self.model = model
        self.version = version
        self.model_dir = os.path.join(os.path.basename(self.models_dir), self.model)
        self.config = config
        self.config.set_task(self.model, self.version)
        self.config.set_test_ids()
        self.processors = processors or {'msa': MsaProcessor,
                                         'fasta': FastaProcessor,
                                         'samcc_test': SamccTestProcessor,
                                         'mutator': None,
                                         'importance_describer': ImportanceDescriber}
        if not os.path.exists(self.models_dir):
            raise ValueError(f"Model path {self.model_dir}. Available:"
                             f"{set(os.listdir(self.models_dir)).difference({'common'})}")
        self.infra = infra

    def predict(self, data, with_model=False, **kwargs):
        base = self.model_dir.replace(os.sep, '.')
        predict = importlib.import_module(f"hamp_pred.src.{base}.predict")
        conf = self.config.dump()
        conf |= kwargs
        for kw in kwargs:
            if hasattr(conf['operator'], kw):
                setattr(conf['operator'], kw, kwargs[kw])
        result = predict.run(data, conf)
        if with_model:
            return result
        return result[0]

    def train(self, data, ids=None, val_ids=None):
        base = self.model_dir.replace(os.sep, '.')
        self.config.operator.parallel = True
        self.config.set_val_ids(val_ids)
        self.config.set_ids(ids)
        train = importlib.import_module(f"hamp_pred.src.{base}.train")
        return train.run(data, self.config.dump())

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
