import os
import atomium
import pandas as pd
import numpy as np


AA_1_to_3 = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
}
AA_3_to_1 = {v.upper() : k.upper() for k, v in AA_1_to_3.items()}


def read_chain_letters(path, chain, start, stop):
    '''
    read .pdb - chain file and returns sequence span from start stop
    '''
    m = atomium.open(path)
    chain_model = m.model.chain(chain)
    residues = [i.name for i in chain_model.residues()]
    if len(residues[0]) == 3:
        residues = [AA_3_to_1[res] for res in residues]
    residues = ''.join(residues)
    pdb_list = [int(i.id.split('.')[-1]) for i in chain_model.residues()]
    start_index, stop_index = pdb_list.index(start), pdb_list.index(stop)
    
    return residues[start_index:stop_index]


def find_mutation(seq1, seq2):
    
    mutation_list = []
    for i, (aa1, aa2) in enumerate(zip(seq1, seq2)):
        if aa1 != aa2:
            mutation_list.append(f'{aa1}{i}{aa2}')
    return mutation_list
        
    
def diffangle(targetA, sourceA):

    a = targetA - sourceA
    a = (a + 180) % 360 - 180
    return a


def find_subsequence_pdb_index(sequence, subsequence):
    """
    search for subsequence in sequence
    where both are list of strings
    returns:
        subsequence_index (list) sequence indices of subsequence
    """
    subsequence_index = list()
    subsequence_iter = iter(subsequence)
    pdb_idx_sub = next(subsequence_iter)
    
    for idx, pdb_idx in enumerate(sequence):
        if pdb_idx == pdb_idx_sub:
            subsequence_index.append(idx)
            try:
                pdb_idx_sub = next(subsequence_iter)
            except StopIteration:
                break
        else:
            pdb
         
    return subsequence_index

def read_scores(path):
    if not os.path.isfile(path):
        raise FileNotFoundError
    df = pd.read_csv(path,
                     delim_whitespace=True,
                     header=0,
                     skiprows=1,
                     index_col=False,
                     engine='c',
                     dtype={"total_score": np.float64})
    df.set_index('description', inplace=True)
    return df

def read_seq_from_pdb(path, chain):
    file = atomium.open(path)
    file = file.model.chain(chain)
    residues = [r.name for r in file.residues()]
    if len(residues[0]) > 1:
        residues = [AA_3_to_1[r] for r in residues]
    return residues


def read_pdb_indices(path, chain):
    
    pdb_list = [s.id.split('.')[1] for s in atomium.open(path).model.chain(chain).residues()]
    pdb_list = [int(idx) for idx in pdb_list]
    
    return pdb_list
