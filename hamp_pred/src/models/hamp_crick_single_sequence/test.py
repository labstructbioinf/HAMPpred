from hamp_pred.src.models.hamp_crick_single_sequence.adjust_prediction import PredictionAdjust
from hamp_pred.src.models.hamp_rot.test import Tester as RotTester, Metrics


class Tester(RotTester):
    def __init__(self, out_column='predicted_rotation', ignored_vals=None):
        super().__init__(out_column=out_column, ignored_vals=ignored_vals or {1000})

    def get_squences(self, data):
        return list([x for x in data['full_sequence']])

    def get_crick_f1(self, test, prediction):
        adjuster = PredictionAdjust()
        prediction['true'] = test['full_crick'].values
        rotation = prediction[self.out_column].notnull().sum() / len(prediction)
        correct = adjuster.get_with_f1_depend(prediction)
        tr, pr, pos_tr, pos_pr = adjuster.get_crick_tr_pred(correct)
        metrics = {'f1_coverage': len(correct) /len(prediction),
                   'rotation_coverage': rotation,
                   'mean_crick_true': tr,
                   'mean_crick_pred': pr,
                   'pos_crick_true': pos_tr,
                   'pos_crick_pred': pos_pr,
                   'mean_crick_mse': Metrics.mse(tr, pr, ignore=self.ignored_vals),
                   'pos_crick_mse': Metrics.mse(pos_tr, pos_pr, ignore=self.ignored_vals)}
        return metrics

    def get_metrics(self, test, prediction):
        rot = super().get_metrics(test, prediction)
        crick = self.get_crick_f1(test, prediction)
        return {**rot, **crick}
