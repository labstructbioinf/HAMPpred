import subprocess

import h5py.h5a
import numpy as np
from Bio.PDB.Polypeptide import aa1
from sklearn.preprocessing import MinMaxScaler

from external.SequenceEncoding.SequenceEncoding import SequenceEncoding, get_dict


class SequenceEncoder:
    def encode(self, seq, *args, **kwargs):
        pass

    def encode_many(self, many):
        if hasattr(many, 'columns') and 'sequence' in many.columns:
            many = many['sequence']
        results = []
        k = len(many)
        c = 0
        for seq in many:
            c += 1
            results.append(self.encode(seq))
            print(f'Encoded {c}/{k} sequences')
        return np.array(results)


class LabelEncoder:

    def as_numpy_array(self, labels):
        return np.array([np.array(x) for x in labels])

    def encode(self, labels, *args, **kwargs):
        pass


class RadiousPhobosEncoder(SequenceEncoder):
    radius = {
        'A': 88.6,
        'R': 173.4,
        'G': 60.1,
        'S': 89.0,
        'C': 108.5,
        'D': 111.1,
        'P': 112.7,
        'N': 114.1,
        'T': 116.1,
        'E': 138.4,
        'Q': 143.8,
        'H': 153.2,
        'M': 162.9,
        'I': 166.7,
        'L': 166.7,
        'K': 168.6,
        'F': 189.9,
        'Y': 193.6,
        'W': 227.8,
        'V': 140.0
    }
    phobos = {
        'I': 4.5,
        'V': 4.2,
        'L': 3.8,
        'F': 2.8,
        'C': 2.5,
        'M': 1.9,
        'A': 1.8,
        'G': -0.4,
        'T': -0.7,
        'S': -0.8,
        'W': -0.9,
        'Y': -1.3,
        'P': 1.6,
        'H': -3.2,
        'D': -3.5,
        'E': -3.5,
        'N': -3.5,
        'Q': -3.5,
        'K': -3.9,
        'R': -4.5
    }

    def __init__(self):
        self.phobos_scaled = self._scale_dict(self.phobos)
        self.radius_scaled = self._scale_dict(self.radius)

    def _scale_dict(self, d):
        a = np.asarray(list(d.values())).reshape(-1, 1)
        v = MinMaxScaler().fit_transform(a)
        v = v.reshape(len(d)).tolist()
        return dict(zip(d.keys(), v))

    def encode(self, seq, *args, **kwargs):
        return np.asarray(list([(self.radius_scaled.get(i, 0), self.phobos_scaled.get(i, 0)) for i in list(seq)]))


class OneHotEncoderSeq(SequenceEncoder):
    def __init__(self, categories='', null_cat='?'):
        cats = categories if categories else aa1
        self._set_params(cats, null_cat)

    def _set_params(self, cats, null_char=''):
        self.cat_dict = {}
        self.cats = cats
        self.null_char = null_char
        for n, c in enumerate(cats + null_char):
            self.cat_dict[c] = n
        self.inv_cat_dict = {val: key for key, val in self.cat_dict.items()}

    def fit(self, seqs):
        dw = set()
        for seq in seqs:
            dw.union(set(seq))
        self._set_params(''.join(list(dw)))

    def encode(self, seq, *args, **kwargs):
        results = []
        for aa in seq:
            dd = [0] * len(self.cats + self.null_char)
            if aa in self.cat_dict:
                pos = self.cat_dict[aa]
                dd[pos] = 1
            results.append(dd)
        return np.array(results)

    def invert(self, enc):
        c = ''
        for row in enc:
            if np.max(row) > 0:
                res = np.argmax(row)
                c += self.inv_cat_dict[res]
            else:
                c += self.null_cat
        return c


class MixedEncoder(SequenceEncoder):
    def __init__(self, types=None):
        self.types = types or ['One_hot']
        self.encoder = SequenceEncoding
        self.enc_dict = {}

    def get_all_types(self):
        return self.encoder.encoding_types

    def set_types(self, types):
        self.types = types
        for _type in self.types:
            self.enc_dict |= get_dict(_type)

    def encode(self, seq, *args, **kwargs):
        results = []
        for t in self.types:
            tmp = [list(x.values())[0] for x in self.encoder(t, self.enc_dict).get_encoding(seq)]
            if len(tmp) == len(seq):
                results.append(tmp)
        return np.concatenate(results, axis=-1)


class RadianEncoder(LabelEncoder):
    def __init__(self, scale=1):
        self.scale = scale

    def encode(self, labels, *args, **kwargs):
        labels = self.as_numpy_array(labels)
        print(labels)
        return np.concatenate([np.sin(np.deg2rad(labels)), np.cos(np.deg2rad(labels))], axis=-1)

    def invert(self, enc):
        labels = self.as_numpy_array(enc)
        res = []
        for row in labels:
            res.append((np.arctan2(row[:, 0], row[:, 1]) * 180 / np.pi).reshape(len(row), 1))
        ret = np.asarray(res)
        return ret


class SeqveqEncoder:
    def __init__(self, seqveq_script='embed_sequences.py'):
        self.seqveq_script = seqveq_script

    def encode(self, sequence, id_=None):
        return self.encode_many([sequence], [id_] if id_ else None)

    def encode_many(self, sequences, ids=None):
        ids = ids or []
        with open('seq.fa', 'w') as seqs:
            for n, seq in enumerate(sequences):
                seqs.write(f'>{ids[n] if ids else n}\n{seq}')

        proc = subprocess.Popen([f'python {self.seqveq_script} --pool avg -o data/demo.h5 seq.fa'])
        proc.communicate()
        with h5py.File('data/demo.h5', "r") as f:
            a_group_key = list(f.keys())[0]
            data = list(f[a_group_key])
        return data


class MultiEncoder(SequenceEncoder):
    def __init__(self, encoders):
        self.encoders = encoders

    def encode(self, seq, *args, **kwargs):
        results = []
        for enc in self.encoders:
            results.append(enc.encode(seq, *args, **kwargs))
        return np.concatenate(results, axis=-1)


def seqs_from_dataframe(dataframe):
    return list(dataframe['sequence'])
