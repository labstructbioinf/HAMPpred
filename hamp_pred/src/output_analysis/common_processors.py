import numpy as np
import pandas as pd
from Bio.PDB.Polypeptide import three_to_one

from hamp_pred.src.input_prep.msa import Msa


class MsaProcessor:
    def __init__(self, path,  model=None):
        self.path = path
        self.name = 'msa'

    @property
    def msa(self):
        return Msa(self.path)

    def prepare_pred(self, data=None):
        return self.msa.sequences

    def prepare_train(self):
        raise NotImplementedError()

    def prepare_out(self, results):
        pred = np.array([np.array(xi) for xi in results['prediction'].values])
        return self.msa.apply_results_on(pred)


class FastaProcessor:
    def __init__(self, path, model=None):
        self.path = path
        self.name = 'fasta'

    def prepare_pred(self, data=None):
        from Bio import SeqIO
        rec = []
        for record in SeqIO.parse(self.path, "fasta"):
            rec.append(str(record.seq))
        return rec


class SamccTestProcessor:
    def __init__(self, path, test_ids=None, n_chains=(0, 1), identifier='source',
                 params=('crdev',), model=None):
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

    def prepare_pred(self, data=None):
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