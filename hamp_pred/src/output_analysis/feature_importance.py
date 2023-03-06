from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB.Polypeptide import aa1

from hamp_pred.src.output_analysis.metrics import Metrics


class ModelMetrics:
    def __init__(self, model, correct_data=None, res_col='N_pred', tr_col='rot'):
        self.model = model
        self.correct_data = correct_data
        self.res_col = res_col
        self.tr_col = tr_col

    def mse(self, data):
        results = data
        mn = self.correct_data[self.tr_col].apply(lambda x: (x[0::2] + x[1::2]) / 2)
        l = np.concatenate(results[self.res_col].values)
        r = np.concatenate(mn.values).reshape(l.shape)
        return np.mean((l - r) ** 2)

    def prepare_out(self, data):
        return {'mse': self.mse(data)}

    def prepare_pred(self, data):
        self.correct_data = data.copy()
        data['n_seq'] = data['n_seq'].apply(lambda x: x[1:-1])
        data['c_seq'] = data['c_seq'].apply(lambda x: x[1:-1])
        seq = []
        for n, r in data.iterrows():
            seq.append(r['n_seq'] + r['c_seq'])
        return seq


class ImportanceDescriber:
    def __init__(self, res_col='N_pred',
                 out_col='importance',
                 out_kind='data',
                 stats_col='diff',
                 model=None, mode='permutation',
                 diff_metric='mse',
                 common_seq=True):
        self.metric = diff_metric
        self.res_col = res_col
        self.out_col = out_col
        self.kind = out_kind
        self.model = model
        self.stats_col = stats_col
        self.mode = mode
        self.common_seq = common_seq

    def permutation_feature_importance(self, out):
        result = out
        seqs = [x for x in result['sequence']]
        if self.common_seq:
            seqs = self._generate_common_sequence(seqs)
            result = self.model.predict(seqs)
        longest = max(seqs, key=len)
        data = {'source_aa': [], 'pos': [], 'seq_id': [], 'seq': [], 'target_aa': [], 'target_pos': []}
        for pos in range(len(longest)):
            for nn, seq in enumerate(seqs):
                seq = seq + ('X' * (len(longest) - len(seq)))
                h_size = len(seq) // 2
                for i in range(len(seq)):
                    if (pos >= h_size and i >= h_size) or (pos < h_size and i < h_size):
                        if i != pos and seq[pos] != seq[i]:
                            ll = list(seq)
                            cw = ll[pos]
                            ll[pos] = ll[i]
                            ll[i] = cw
                            data['seq'].append(''.join(ll))
                            data['seq_id'].append(nn)
                            data['pos'].append(pos)
                            data['source_aa'].append(seq[pos])
                            data['target_aa'].append(seq[i])
                            data['target_pos'].append(seq[i])
        data = pd.DataFrame(data)
        results = self.model.predict([x for x in data.seq])[[self.res_col]]
        data['new_pred'] = results[self.res_col]
        result['seq_id'] = range(len(result))
        data = pd.merge(result, data, on=['seq_id']).drop(columns=['seq'])
        data['diff'] = data.apply(lambda x: self._apply_metric(x), axis=1)
        data['pos_diff'] = data.apply(lambda x: Metrics.mse_f1(x[self.res_col][max(x['pos'] - 2, 0):x['pos'] + 2],
                                                               x['new_pred'][max(x['pos'] - 2, 0):x['pos'] + 2]),
                                      axis=1)
        data.dropna(subset=['diff'], inplace=True)
        data['pos_diff'] = data['pos_diff'].fillna(0)
        per_seq = data.groupby(['seq_id', 'pos', 'source_aa'], as_index=False). \
            agg({'diff': 'mean', 'pos_diff': 'mean'}).sort_values(by=['seq_id', 'pos', 'diff'],
                                                                  ascending=[True, True, False])
        per_seq.set_index(['seq_id'], inplace=True, drop=False)
        total = data.groupby(['pos', 'source_aa'], as_index=False).agg(
            {'diff': 'mean', 'pos_diff': 'mean'}).sort_values(by=['pos', 'diff'],
                                                              ascending=[True,
                                                                         False])
        mutations = data.groupby(['seq_id', 'pos', 'source_aa', 'target_aa'], as_index=False). \
            agg({'diff': 'mean'}).sort_values(by=['seq_id', 'pos', 'diff'],
                                              ascending=[True, True, False])
        return total, per_seq, data, mutations

    def _generate_sequences(self, seq, seq_id, positions_dict=None, padd=0):
        positions_dict = positions_dict or {}
        l_seq = list(seq)
        data = {'source_aa': [], 'pos': [], 'seq_id': [], 'seq': [], 'target_aa': []}
        for pos, aa in enumerate(l_seq):
            for repl in positions_dict.get(pos, aa1):
                prv = l_seq[pos]
                if prv == repl:
                    continue
                l_seq[pos] = repl
                if padd:
                    new_seq = ''.join(l_seq) + 'X' * padd
                else:
                    new_seq = ''.join(l_seq)
                data['seq'].append(new_seq)
                data['seq_id'].append(seq_id)
                data['pos'].append(pos)
                data['source_aa'].append(seq[pos])
                data['target_aa'].append(repl)
                l_seq[pos] = prv
        return pd.DataFrame(data)

    def _generate_common_sequence(self, seqs):
        pos_dict = defaultdict(lambda: defaultdict(int))
        longest = max(seqs, key=len)
        for seq in seqs:
            new_seq = seq + (len(longest) - len(seq)) * 'X'
            for n, aa in enumerate(new_seq):
                pos_dict[n][aa] += 1
        seq = ''
        for pos, res in pos_dict.items():
            seq += max(res, key=res.get)
        common_seqs = []
        for pos, res in pos_dict.items():
            for aa, cnt in res.items():
                new_seq = seq[:pos] + aa + seq[pos + 1:]
                common_seqs.append(new_seq)
        return list(set(common_seqs))

    def replacement_feature_importance(self, out):
        seqs = [x for x in out['sequence']]
        longest = max(seqs, key=len)
        if self.common_seq:
            seqs = self._generate_common_sequence(seqs)
            out = self.model.predict(seqs)
        out['seq_id'] = range(len(out))
        new_seqs = []
        for pos, seq in enumerate(seqs):
            new_seqs.append(self._generate_sequences(seq, pos, padd=len(longest) - len(seq)))
        data = pd.concat(new_seqs)
        result = self.model.predict([x for x in data.seq])[[self.res_col]]
        data['new_pred'] = result[self.res_col]
        data = pd.merge(data, out, on=['seq_id'])
        data['diff'] = data.apply(lambda x: self._apply_metric(x), axis=1)
        per_seq = data.groupby(['seq_id', 'pos', 'source_aa'], as_index=False). \
            agg({'diff': 'mean'}).sort_values(by=['seq_id', 'pos', 'diff'],
                                              ascending=[True, True, False])
        per_seq.set_index(['seq_id'], inplace=True, drop=False)
        total = data.groupby(['pos', 'source_aa'], as_index=False).agg(
            {'diff': 'mean'}).sort_values(by=['pos', 'diff'], ascending=[True, False])
        mutations = data.groupby(['seq_id', 'pos', 'source_aa', 'target_aa'], as_index=False). \
            agg({'diff': 'mean'}).sort_values(by=['seq_id', 'pos', 'diff'],
                                              ascending=[True, True, False])
        return total, per_seq, data, mutations

    def prepare_out(self, out):
        if self.kind == 'data':
            return getattr(self, f'{self.mode}_feature_importance')(out)
        elif self.kind == 'mutations':
            return getattr(self, f'{self.mode}_feature_importance')(out)[-1]
        elif self.kind == 'plot_seq':
            return self.plot_importance_per_seq(out)
        elif self.kind == 'heatmap':
            return self.to_heatmap(out)

    def plot_importance_per_seq(self, out, limit=20):
        total, per_seq, data, mutations = getattr(self, f'{self.mode}_feature_importance')(out)
        g = sns.FacetGrid(per_seq, row="seq_id")
        g.map(sns.barplot, "source_aa", 'diff')
        return g

    def to_heatmap(self, out):
        res = []
        one_seq = False
        out, per_seq, data, mutations = getattr(self, f'{self.mode}_feature_importance')(out)
        if len(per_seq.seq_id.unique()) == 1:
            one_seq = True
        if one_seq:
            return np.array(
                [np.array(out[self.stats_col].values / out[self.stats_col].sum()) for i in range(len(out))]), list(
                out['source_aa'].values)
        for ind, group in out.groupby('pos'):
            aa_pos = []
            for aa in aa1:
                r = group[group['source_aa'] == aa][self.stats_col]
                if not r.empty:
                    aa_pos.append(r.iloc[0])
                else:
                    aa_pos.append(0)
            aa_pos = np.array(aa_pos) / sum(np.array(aa_pos))
            res.append(aa_pos)
        return np.array(res), list(aa1)

    def _apply_metric(self, x):
        if self.metric == 'mse':
            return Metrics.mse_f1(x[self.res_col], x['new_pred'])
        elif self.metric == 'signed_mean':
            return np.mean(x['new_pred']) - np.mean(x[self.res_col])
        elif self.metric == 'unsigned_mean':
            return np.abs(np.mean(x[self.res_col]) - np.mean(x['new_pred']))
        else:
            raise ValueError(f"Unknown metric {self.metric}")