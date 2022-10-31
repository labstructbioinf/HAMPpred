import numpy as np

from hamp_pred.src.output_analysis.metrics import Metrics


class Tester:
    def __init__(self, out_column='prediction', ignored_vals=None, scale=2):
        self.out_column = out_column
        self.ignored_vals = ignored_vals
        self.scale = scale

    def get_metrics(self, test, prediction):
        mean_rot_true, mean_pred = [], []
        pos_rot_true, pos_rot_pred = [], []

        for n, (ind, row) in enumerate(test.iterrows()):
            n_rot, c_rot = row['n_crick_mut'], row['c_crick_mut']
            rot = n_rot - c_rot
            rot = (rot + 180) % 360 - 180
            rot = (rot[::2] + rot[1::2]) / 2
            pred = prediction.iloc[n][self.out_column]
            if pred is None:
                continue
            mean_rot_true.append(np.mean(rot))
            pos_rot_true.extend(rot)
            pred = list(np.reshape(pred, len(pred)))
            mean_pred.append(np.mean(pred))
            pos_rot_pred.extend(pred)
        metrics = {'true_pos_rot': pos_rot_true,
                   'pred_pos_rot': pos_rot_pred,
                   'true_mean_rot': mean_rot_true,
                   'pred_mean_rot': mean_pred}
        return self._to_arrays(metrics)

    def get_squences(self, data):
        sequences, mean_rot = [], []
        for ind, row in data.iterrows():
            seq = row['n_seq'][1:-1] + row['c_seq'][1:-1]
            sequences.append(seq)
        return sequences

    def _to_arrays(self, metrics):
        for key, value in metrics.items():
            if isinstance(value, list) or isinstance(value, tuple):
                metrics[key] = np.array(value) / self.scale
        metrics['mse_seq'] = Metrics.mse(metrics['true_mean_rot'], metrics['pred_mean_rot'], ignore=self.ignored_vals)
        metrics['mse_pos'] = Metrics.mse(metrics['true_pos_rot'], metrics['pred_pos_rot'], ignore=self.ignored_vals)
        return metrics
