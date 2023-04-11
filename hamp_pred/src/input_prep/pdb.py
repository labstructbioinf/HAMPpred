import os
from collections import defaultdict

import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import d3_to_index, dindex_to_1
#from samcc.bundleClass import bundleClass


def measure_one_HAMP(path_hamp, a1_start=None, a1_stop=None, a2_start=None, a2_stop=None, chain1=None, chain2=None):
    '''
    calculate HAMP protein descriptors
    params:
        path_hamp (str) path to .pdb file
        a1_start (int) 1st chain hamp motif pdb start index
        a1_stop (int) 1st chain hamp motif pdb stop index
        a2_start (int) 2nd chain hamp motif pdb start index
        a2_stop (int) 2nd chain hamp motif pdb stop index
        chain1 (str) one letter pdb 1st chain name
        chain2 (str) one letter pdb 2nd chain name
    return:
        bundle_df (pd.DataFrame) detailed description of HAMP
        n_crick (np.array) N crick angles per residue, also in bundle_df
        c_crick (np.array) C crick angles per residue, also in bundle_df
    '''
    chain1_range = range(a1_start, a1_stop)
    chain2_range = range(a2_start, a2_stop)

    defdata = [
        [chain1_range, chain2_range, chain1_range, chain2_range],
        [chain1, chain1, chain2, chain2],
        [False, False, False, False],  # False for parallel orientation
        ['x', 'x', 'x', 'x']
    ]

    bundle = bundleClass()
    bundle.from_defdata(path_hamp, *defdata)
    bundle.calc_bundleaxis()
    bundle.calc_crick()
    bundle.calc_radius()
    bundle.calc_periodicity()
    bundle.calc_crickdev(P=3.5, REP=7, optimal_ph1=19.5)
    bundle_df = bundle.gendf()

    crick = bundle_df.crick.values
    n_crick = crick[0::2]
    c_crick = crick[1::2]

    return bundle_df, n_crick, c_crick, crick


class PdbDataProcessor:
    def __init__(self, model=None, auto=False,
                 a1_start=None, a1_stop=None, a2_start=None, a2_stop=None, chain1=None, chain2=None,
                 ap=True):
        self.auto = auto
        self.model = model

        self.a1_start = a1_start
        self.a1_stop = a1_stop
        self.a2_start = a2_start
        self.a2_stop = a2_stop
        self.chain1 = chain1
        self.chain2 = chain2
        self.ap = ap

    def prepare_train(self, folder, a1_start=None, a1_stop=None, a2_start=None, a2_stop=None, chain1=None, chain2=None,
                      ap=False):
        files = []
        if os.path.exists(folder):
            files = [os.path.join(folder, f) for f in os.listdir(folder)]
        a1_start = a1_start or self.a1_start
        a1_stop = a1_stop or self.a1_stop
        a2_start = a2_start or self.a2_start
        a2_stop = a2_stop or self.a2_stop
        chain1 = chain1 or self.chain1
        chain2 = chain2 or self.chain2
        ap = ap or self.ap
        dataset = {'n_crick_mut': [], 'c_crick_mut': [], 'n_seq': [], 'c_seq': [], 'full_sequence': [],
                   'full_crick': []}
        for ff in files:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('temp', ff)
            bundle, n_cr, c_cr, crick = measure_one_HAMP(ff, a1_start, a1_stop, a2_start, a2_stop, chain1, chain2)
            sequence_hamp = ''.join([dindex_to_1[d3_to_index[x]] for x in bundle.res_name.values])
            dataset['n_seq'].append(sequence_hamp[0::4])
            dataset['c_seq'].append(sequence_hamp[1::4])
            dataset['n_crick_mut'].append(n_cr[2:-2])
            dataset['c_crick_mut'].append(c_cr[2:-2])
            n_cr_mean = (n_cr[2:-2][0::2] + n_cr[2:-2][1::2]) / 2
            c_cr_mean = (c_cr[2:-2][0::2] + c_cr[2:-2][1::2]) / 2

            sequence, _ = self._get_sequence(structure, [chain1, chain2], ap=ap)
            sequence = ''.join(sequence)
            n_st, c_st = sequence.find(sequence_hamp[0::4]), sequence.find(sequence_hamp[1::4])
            full_hamp_seq = sequence[n_st:c_st + len(sequence_hamp[1::4])]
            full_crick = [10000] * len(full_hamp_seq)
            full_crick[1:1 + len(n_cr_mean)] = n_cr_mean
            full_crick[c_st + 1:c_st + len(c_cr_mean) + 1] = c_cr_mean
            dataset['full_crick'] = sequence
            dataset['full_sequence'] = full_hamp_seq
        return pd.DataFrame(dataset)

    def prepare_pred(self, folder, a1_start=None, a1_stop=None, a2_start=None, a2_stop=None, chain1=None, chain2=None,
                     ap=True):
        return self.prepare_train(folder, a1_start, a1_stop, a2_start, a2_stop, chain1, chain2, ap)

    def _get_sequence(self, structure, chains=None, ap=True, rr=()):
        chain_residues = defaultdict(list)
        all_sequence = []
        parr = 0
        c = 0
        for chain in structure[0]:
            sequence = []
            if chains and chain.id not in chains:
                continue
            for residue in chain:
                r = rr[c] if rr else None
                if r and (residue.id[1] < r[0] or residue.id[1] > r[1]):
                    continue
                aa = dindex_to_1[d3_to_index[residue.resname]]
                sequence.append(aa)
                chain_residues[chain].append(aa)
            c += 1
            if ap:
                if parr % 2:
                    sequence = sequence[::-1]
                parr += 1
            all_sequence.extend(sequence)
        return all_sequence, chain_residues
