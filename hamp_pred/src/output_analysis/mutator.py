import bisect
import itertools
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
from Bio.PDB.Polypeptide import aa1


class HeptadMutator:
    def __init__(self, length=7, helices=2, unique_per_helix=False):
        self.alphabet = aa1
        self.length = length
        self.helices = helices
        self.unique_per_helix = unique_per_helix
        self._load_fragments()

    def _load_fragments(self):
        path = pathlib.Path(__file__).parent.parent.parent.parent.joinpath('data/output/mutator')
        with open(path.joinpath('neg_fragments.p'), 'rb') as r:
            self._neg = pickle.load(r)
        with open(path.joinpath('neg_fragments.p'), 'rb') as r:
            self._pos = pickle.load(r)

    @property
    def all_combinations(self):
        if self.unique_per_helix:
            return list(itertools.combinations_with_replacement(self.alphabet, self.length * self.helices))
        else:
            result = list(itertools.combinations_with_replacement(self.alphabet, self.length))
            mocked = ['X' * self.length] * len(result)
            return [''.join(a) + b for a, b in zip(result, mocked)]

    def prepare_pred(self, sample=10000):
        return random.sample(self.all_combinations, sample, counts=[1] * len(self.all_combinations))

    def analyze(self, out: pd.DataFrame):
        out['N_mean'] = out['N_pred'].apply(lambda x: np.mean(x))
        out.sort_values(by='N_mean', inplace=True)
        out['is_plus'] = out['N_mean'] >= 0
        return out

    def prepare_out(self, out):
        return self.analyze(out)

    def find_closest_sequence_with_crdev(self, sequence, crdev=10, space_size=1000):
        if crdev < 0:
            dd = {k: v for k, v in sorted(self._neg.items(), key=lambda item: item[1])}
        else:
            dd = {k: v for k, v in sorted(self._pos.items(), key=lambda item: item[1])}
        pos = bisect.bisect_left(list(dd.values()), crdev)
        seqs = list(dd.keys())[max(pos - space_size, 0):pos + space_size]
        return self._closest_search_brute_force(sequence, seqs)

    def _closest_search_brute_force(self, to_modify, seqs):
        blocks = []
        for i in range(0, len(to_modify) - self.length + 1, self.length):
            mx = 0
            for frag in seqs:
                df = self._diff(to_modify[i:i + self.length], frag)
                if df >= mx:
                    mx = df
                    fr = frag
            blocks.append(fr)
        return ''.join(blocks)

    def _diff(self, seq1, seq):
        sc = 0
        for i, j in zip(seq1, seq):
            if i == j:
                sc += 1
        return sc


