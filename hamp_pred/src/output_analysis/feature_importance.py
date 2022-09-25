import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ImportanceDescriber:
    def __init__(self, res_col='N_pred',
                 out_col='importance',
                 out_kind='data',
                 model=None):
        self.res_col = res_col
        self.out_col = out_col
        self.kind = out_kind
        self.model= model

    def mse(self, tr, exp):
        sm = 0
        for t, e in zip(tr, exp):
            sm += (t - e) ** 2
        res = sm / len(exp)
        return res if isinstance(res, float) else res[0]

    def feature_importance(self, out):
        result = out
        seqs = [x for x in result['sequence']]
        data = {'source_aa': [], 'pos': [], 'seq_id': [], 'seq': [], 'target_aa': []}
        for pos in range(len(seqs[0])):
            for nn, seq in enumerate(seqs):
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
        data = pd.DataFrame(data)
        results = self.model.predict([x for x in data.seq])[[self.res_col]]
        data['new_pred'] = results[self.res_col]
        result['seq_id'] = range(len(result))
        data = pd.merge(result, data, on=['seq_id']).drop(columns=['seq'])
        data['diff'] = data.apply(lambda x: self.mse(x['N_pred'], x['new_pred']), axis=1)
        per_seq = data.groupby(['seq_id', 'pos', 'source_aa'], as_index=False). \
            agg({'diff': 'mean'}).sort_values(by=['seq_id', 'pos', 'diff'], ascending=[True, True, False])
        per_seq.set_index(['seq_id'], inplace=True, drop=False)
        total = data.groupby(['pos', 'source_aa'], as_index=False).agg({'diff': 'mean'}).sort_values(by=['pos', 'diff'],
                                                                                                     ascending=[True,
                                                                                                                False])
        return total, per_seq

    def prepare_out(self, out):
        if self.kind == 'data':
            return self.feature_importance(out)
        elif self.kind == 'plot_seq':
            return self.plot_importance_per_seq(out)
        else:
            self.plot_importance_overall(out)

    def plot_importance_per_seq(self, out, limit=20):
        total, per_seq = self.feature_importance(out)
        g = sns.FacetGrid(per_seq, row="seq_id")
        g.map(sns.barplot, "source_aa", 'diff')
        return g

    def plot_importance_overall(self, out):
        total, per_seq = self.feature_importance(out)
        # for ind, row in per_seq.iterrows():
