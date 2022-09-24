import importlib
import os

import numpy as np
import pandas as pd
from Bio.PDB.Polypeptide import three_to_one
from keras import backend as K

from hamp_pred.src.input_prep.msa import Msa
from hamp_pred.src.output_analysis.mutator import HeptadMutator
from hamp_pred.src.predictor_config import DEFAULT_CONF


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
                                         'mutator': None}
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

    def process_data(self, *args, kind='msa', **kwargs):
        processor = self.processors.get(kind)(*args, **kwargs)
        to_pred = processor.prepare_pred()
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


class MsaProcessor:
    def __init__(self, path):
        self.path = path
        self.name = 'msa'

    @property
    def msa(self):
        return Msa(self.path)

    def prepare_pred(self):
        return self.msa.sequences

    def prepare_train(self):
        raise NotImplementedError()

    def prepare_out(self, results):
        pred = np.array([np.array(xi) for xi in results['prediction'].values])
        return self.msa.apply_results_on(pred)


class FastaProcessor:
    def __init__(self, path):
        self.path = path
        self.name = 'fasta'

    def prepare_pred(self):
        from Bio import SeqIO
        rec = []
        for record in SeqIO.parse(self.path, "fasta"):
            rec.append(str(record.seq))
        return rec


class SamccTestProcessor:
    def __init__(self, path, test_ids=None, n_chains=(0, 1), identifier='source',
                 params=('crdev',)):
        self.path = path
        self.test_ids = set(test_ids) if test_ids else None
        self.n_chains = n_chains
        self.identifier = identifier
        self.params = params
        self.name = 'samcc_test'
        self._load(path)

    def adjust_config(self, config):
        config.operator.parallel = True
        return config

    def _load(self, path):
        if path.endswith('.tsv'):
            dd = pd.read_csv(path, sep='\t')
        else:
            dd = pd.read_pickle(path)
        if self.identifier in dd.columns:
            dd = dd.set_index(self.identifier)
            if self.test_ids:
                dd = dd.loc[self.test_ids]
        else:
            raise ValueError(f'Did not found identifier {self.identifier}')
        self.data = dd
        return dd

    def get_data(self, seq_sep=''):
        Xx, Yy = [], []
        c = 0
        identifiers = []
        for gr, data in self.data.groupby(by=self.identifier, level=0):
            X, Y = [], []

            for chain in self.n_chains:
                N = data[data['chain'] == chain]
                Nx = N['res_name']
                Nx = ''.join([three_to_one(x) if len(x) != 1 else x for x in Nx])
                Ny = list(N[list(self.params)].values)
                X.append(Nx)
                Y.extend(Ny)
                c += 1
            identifiers.append(gr)
            X = seq_sep.join(X)
            Xx.append(X)
            Yy.append(np.array(Y))
        return Xx, Yy, identifiers

    def prepare_pred(self):
        x, y, id_ = self.get_data()
        return x

    def prepare_out(self, results):
        x, y, id_ = self.get_data()
        results['true_value'] = y
        return results

    def to_fasta(self, ff='samcc_seq', combine_chains=True):
        x, y, ids_ = self.get_data(' ')
        cc = 1 if combine_chains else len(self.n_chains)
        chains = [[] for _ in range(cc)]
        for n, seq in enumerate(x):
            ch = ''
            for c, chain in enumerate(seq.split(' ')):
                if combine_chains:
                    ch += chain
                else:
                    chains[c].append(f'>{ids_[n]}\n{chain}')
            if ch:
                chains[0].append(f'>{ids_[n]}\n{ch}')
        for n, p in enumerate(chains):
            with open(ff + '_' + str(n) + '.fasta', 'w') as pc:
                pc.write('\n'.join(p))
