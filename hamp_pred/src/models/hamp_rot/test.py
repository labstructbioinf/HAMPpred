import numpy as np

from hamp_pred.src.output_analysis.metrics import Metrics


class Tester:
    def __init__(self, out_column='prediction', ignored_vals=None):
        self.out_column = out_column
        self.ignored_vals = ignored_vals

    def get_metrics(self, test, prediction):
        mean_rot_true, mean_pred = [], []
        pos_rot_true, pos_rot_pred = [], []

        for n, (ind, row) in enumerate(test.iterrows()):
            n_rot, c_rot = row['n_crick_mut'], row['c_crick_mut']
            n_rot = (n_rot[::2] + n_rot[1::2]) / 2
            c_rot = (c_rot[::2] + c_rot[1::2]) / 2
            pred = prediction.iloc[n][self.out_column]
            if pred is None:
                continue
            mean_rot_true.append(np.mean(n_rot - c_rot))
            pos_rot_true.extend(list(n_rot - c_rot))
            pred = list(np.reshape(pred, len(pred)))
            mean_pred.append(np.mean(pred))
            pos_rot_pred.extend(pred)
        metrics = {'mse_seq': Metrics.mse(mean_rot_true, mean_pred, ignore=self.ignored_vals),
                   'mse_pos': Metrics.mse(pos_rot_true, pos_rot_pred, ignore=self.ignored_vals),
                   'true_pos_rot': pos_rot_true,
                   'pred_pos_rot': pos_rot_pred,
                   'true_mean_rot': mean_rot_true,
                   'pred_mean_rot': mean_pred}
        return metrics

    def get_squences(self, data):
        sequences, mean_rot = [], []
        for ind, row in data.iterrows():
            seq = row['n_seq'][1:-1] + row['c_seq'][1:-1]
            sequences.append(seq)
        return sequences

