import numpy as np
import pandas as pd
from Bio.PDB.Polypeptide import three_to_one

from hamp_pred.src.input_prep.encode import LabelEncoder


class PandasOutput:
    def __init__(self, columns=None):
        self.columns = columns or ['id', 'sequence', 'length']

    def from_list_of_lists(self, data):
        return pd.DataFrame.from_records(data, columns=self.columns)


class SequencePreparator:
    def prepare(self, data):
        pass


class SeqWindow(SequencePreparator):
    def __init__(self, window_size, shift=None, null_char='?', out=None):
        self.window_size = window_size
        self.shift = shift
        self.out = out or PandasOutput()
        self.null_char = null_char

    def prepare(self, seq, id_=0):
        results = []
        pos = id_
        for p in range(0, len(seq), self.shift):
            part = seq[p:p + self.window_size]
            if len(part) != self.window_size:
                results.append((pos,
                                part + (self.null_char * (-len(part) + self.window_size)),
                                len(seq)))
            else:
                results.append((pos, part, len(seq)))
        return results

    def prepare_many(self, many_seqs):
        results = []
        for n, seq in enumerate(many_seqs):
            results.extend(self.prepare(seq, n))
        if self.out:
            return self.out.from_list_of_lists(results)
        return results

    def invert(self, prep, preds=None, resolve_common=None):
        results = []
        preps = []
        c = 0
        for id_, data in prep.groupby('id'):
            res = ''
            lg = data[['length']].iloc[0]['length']
            for seq in data['sequence']:
                if res:
                    res += seq[self.window_size - self.shift:]
                else:
                    res += seq
            if preds is not None:
                cc = preds[c: c + len(data)]
                preps.append(self._resolve_preds(cc, lg, resolve_common))
                c += len(data)
            results.append(res[:lg])
        if preps:
            return results, preps
        return results

    def _resolve_preds(self, preds, seq_length, func=None):
        af = []
        for arr in preds:
            for aa in arr:
                af.append(aa)
        res = []
        for i in range(0, len(af), self.window_size):
            res.extend(af[i:i + self.shift])
            left = af[i + self.shift:i + self.window_size]
            right = af[i + self.window_size: i + self.window_size + (self.window_size - self.shift)]
            diff = self.window_size - self.shift
            if not func:
                res.extend(left[:diff // 2])
                res.extend(right[diff // 2:])
            else:
                res.extend(func(left, right))
        return res[:seq_length]


class MultiChainOperator:
    def __init__(self, encoder, preparator=None,
                 y_encoder=None,
                 y_preparator=None,
                 n_chains=2, chain_names=('N', 'C'),
                 solo_helix=False,
                 sep=' ', auto_split=True,
                 linkers_length=None, lengths=None,
                 parallel=False, first_last_skip=True):
        self.n_chains = n_chains
        self.preparator = preparator
        self.encoder = encoder
        self.y_preparator = y_preparator
        self.y_encoder = y_encoder
        self.sep = sep
        self.solo_helix = solo_helix
        self.auto_split = auto_split
        self.chain_names = chain_names if len(chain_names) == self.n_chains else [str(f'seq_{i}') for i in
                                                                                  range(self.n_chains)]
        self.linkers_length = linkers_length or []
        self.lengths = lengths or []
        self.linker_mark = [0]
        self.parallel = parallel
        self.first_last_skip = first_last_skip
        self._prep_chains = None
        self._data = None

    def _check(self, seq):
        chains = seq.split(self.sep) if isinstance(seq, str) else seq
        if len(chains) != self.n_chains:
            raise ValueError(f'There is {len(chains)}, expected {self.n_chains}')
        lengths = []
        for pos in chains:
            lengths.append(len(pos))
        if isinstance(chains[0], str):
            seq = ''.join(chains)
        else:
            seq = np.concatenate(seq)
        return seq, chains, lengths, self.linkers_length

    def _auto_split(self, seq):
        tt = len(seq) - sum(self.linkers_length)
        rest = tt % self.n_chains
        lgh = tt // self.n_chains
        helises = []
        lengths = self.lengths or [lgh] * self.n_chains
        lks = []
        wc = 0
        for p in range(self.n_chains - 1):
            if wc < rest:
                wc += 1
                lks.append(1)
            else:
                lks.append(0)
        linkers = self.linkers_length or lks
        c = 0
        n = 0
        for ll in lengths:
            helises.append(seq[c:c + ll])
            c += ll
            if n < len(linkers):
                c += linkers[n]
                n += 1
        return helises, lengths, linkers

    def _prepare_prediction(self, helises):
        if not self.parallel:
            for i in range(1, len(helises), 2):
                if isinstance(helises[i], str):
                    helises[i] = helises[i][::-1]
                else:
                    helises[i] = list(reversed(helises[i]))
        return helises

    def _get_for_prediction(self, data, ids=None):
        results = []
        for n, seq in enumerate(data):
            row = []
            if self.auto_split and isinstance(seq, str):
                helises, ll, linkers = self._auto_split(seq)
            else:
                seq, helises, ll, linkers = self._check(seq)
            helises = self._prepare_prediction(helises)
            row.append(n)
            if ids and ids[n]:
                row.append(ids[n])
            else:
                row.append(None)
            row.append(seq)
            for helix in helises:
                row.append(helix)
            row.append(ll)
            row.append(linkers)
            results.append(row)
        cols = ['id', 'external_id', 'sequence']
        for c in self.chain_names:
            cols.append(c)
        cols.extend(['lengths', 'linkers'])
        return pd.DataFrame.from_records(results,
                                         columns=cols)

    def get_for_prediction(self, data, ids=None):
        data = self._get_for_prediction(data, ids)
        chains = []
        for chain in self.chain_names:
            chains.append(self.preparator.prepare_many(data[chain]))
        encoded = []
        for chain in chains:
            encoded.append(self.encoder.encode_many(chain))
        if not self.solo_helix:
            enc = np.concatenate(encoded, axis=-1)
        else:
            enc = encoded
        self._prep_chains = chains
        self._data = data
        return enc

    def get_from_prediction(self, prediction, n_features=1, shrink_factor=1, result_col='prediction'):
        xp = self._data.copy()
        for helix_pos, prep in enumerate(self._prep_chains):
            helix = prediction[:, :, helix_pos * n_features: (helix_pos + 1) * n_features]
            preds, helix = self.preparator.invert(prep, helix)
            if self.y_encoder:
                helix = self.y_encoder.invert(helix)
            xp[self.chain_names[helix_pos] + '_pred'] = list(helix)

        def merge(x):
            res = []
            for n, name in enumerate(self.chain_names):
                res.append(x[name + '_pred'])
                fc = n_features // shrink_factor
                if x['linkers'] and n < len(x['linkers']):
                    lk = x['linkers'][n] * [self.linker_mark * fc]
                    if lk:
                        res.append(lk)
            return np.concatenate(res)

        xp[result_col] = xp.apply(lambda x: merge(x), axis=1)
        return xp

    def get_for_train(self, X, y=None, ids=None,
                      test_ids=None, val_ids=None,
                      test_size=0.1, valid_size=0.1,
                      source='samcc', source_config=None):
        if y is None and source:
            conf = source_config or {}
            if source == 'samcc':
                X, y, ids = self._get_for_train_from_samcc(X, **conf)
            else:
                raise ValueError('Unknown source. Available only: samcc')
        test_idx = None
        if test_ids and ids:
            test_idx = {id_: n for n, id_ in enumerate(ids) if id_ in test_ids}
            test_idx = list(test_idx.values())
        val_idx = None
        if val_ids and ids:
            val_idx = {id_: n for n, id_ in enumerate(ids) if id_ in val_ids}
            val_idx = list(val_idx.values())
        y = self._get_for_prediction(y)
        X = self._get_for_prediction(X, ids)
        chains = []
        labels = []
        for chain in self.chain_names:
            chains.append(self.preparator.prepare_many(X[chain]))
            labels.append(self.y_preparator.prepare_many(y[chain]))
        encoded = []
        encoded_labels = []
        for chain, label in zip(chains, labels):
            encoded.append(self.encoder.encode_many(chain['sequence']))
            labs = label['sequence'].values
            if self.y_encoder:
                labs = self.y_encoder.encode(labs)
            encoded_labels.append(LabelEncoder().as_numpy_array(labs))
        self._prep_chains = chains
        self._data = X
        train, val, test = self._train_valid_test_split(valid_size,
                                                        test_size, test_ids=test_idx,
                                                        val_ids=val_idx)
        if not self.solo_helix:
            enc = np.concatenate(encoded, axis=-1)
            lab = np.concatenate(encoded_labels, axis=-1)

            tr = (enc[train, :, :], lab[train, :, :])
            val = (enc[val, :, :], lab[val, :, :])
            test = (enc[test, :, :], lab[test, :, :])
            return tr, val, test
        en = []
        trX, trY = [], []
        valX, valY = [], []
        testX, testY = [], []
        for enc, lab in zip(encoded, encoded_labels):
            trX.append(enc[train, :, :])
            valX.append(enc[val, :, :])
            testX.append(enc[test, :, :])
            trY.append(lab[train, :, :])
            valY.append(lab[val, :, :])
            testY.append(lab[test, :, :])
        return (trX, trY), (valX, valY), (testX, testY)

    def _train_valid_test_split(self, val_s=0.1, test_s=0.1, test_ids=None, val_ids=None):
        ids = set(self._data['id'])
        tap = [0] * (len(set(ids)) + 1)
        for w, data in self._data.groupby('id'):
            tap[w + 1] = tap[w] + len(data)
        tt = max(ids)
        val_size = int(val_s * tt)
        test_size = int(test_s * tt)
        kp = list(ids)
        np.random.shuffle(kp)
        if not test_ids:
            test_ids = kp[:test_size]
        ids = ids.difference(test_ids)
        kp = list(ids)
        np.random.shuffle(kp)
        if not test_ids:
            val_ids = kp[:val_size]
        train_ids = ids.difference(val_ids)

        tr, vl, te = [], [], []
        for p in train_ids:
            tr.extend(list(range(tap[p], tap[p + 1])))
        for p in test_ids:
            te.extend(list(range(tap[p], tap[p + 1])))
        for p in val_ids:
            vl.extend(list(range(tap[p], tap[p + 1])))
        return tr, te, vl

    def _get_for_train_from_samcc(self, data=None,
                                  processor=None,
                                  params=('crdev',),
                                  chain_map=None,
                                  identifier='source'):
        data = data or data
        chain_map = chain_map or {}
        if isinstance(data, pd.DataFrame):
            data, path = data, None
        else:
            data, path = None, data
        processor = processor or SamCCOutputReader(path, data, struct_identifier=identifier)
        X, Y, IDs = [], [], []
        for n in range(len(self.chain_names)):
            ch = chain_map.get(self.chain_names[n], n)
            x, y, ids = processor.get_by_chain_and_params(ch, params)
            X.append(x)
            Y.append(y)
            IDs.append(ids)
        return list(zip(*X)), list(zip(*Y)), ids


class SamCCOutputReader:
    def __init__(self, path=None, data=None, struct_identifier='source',
                 test_ids=None):
        self.path = path
        self.test_ids = set(test_ids) if test_ids else None
        self.data = data
        if self.data is None and self.path is None:
            raise ValueError('Provide path or data for samcc output')
        self.identifier = struct_identifier
        if self.data is None:
            self._load(self.path)

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

    def get_by_chain_and_params(self, chain, params, limit=None):
        X, Y, ids = [], [], []
        c = 0
        for gr, data in self.data.groupby(by=self.identifier, level=0):
            N = data[data['chain'] == chain]
            Nx = N['res_name']
            Nx = ''.join([three_to_one(x) if len(x) != 1 else x for x in Nx])
            Ny = list(N[list(params)].values)
            X.append(Nx)
            Y.append(Ny)
            ids.append(gr)
            c += 1
            if limit and c > limit:
                break
        return X, Y, ids
