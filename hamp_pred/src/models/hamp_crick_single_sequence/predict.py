from hamp_pred.src.models.common.models import BaseLinearWrapper
from hamp_pred.src.models.hamp_crick_single_sequence.adjust_prediction import PredictionAdjust
from hamp_pred.src.models.hamp_crick_single_sequence.test import Tester


def run(sequences, config=None):
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    adjuster = PredictionAdjust()
    n_chains, features = operator.n_chains, 2
    if config.get('is_test', False):
        tester = Tester()
        test_data = sequences
        sequences = tester.get_squences(sequences)
    to_pred = operator.get_for_prediction(sequences)
    model = model or BaseLinearWrapper(config=config)
    inp_shape = to_pred.shape[1], to_pred.shape[-1]
    md = model.build(inp_shape, features).compile(optimizer="Adam", loss="mse", metrics=["mae"])
    prediction = md.predict(to_pred)
    results = operator.get_from_prediction(prediction, n_features=2, shrink_factor=2), md, to_pred
    data = results[0]
    data = adjuster.flatten_prediction(data)
    segment_info = data['prediction'].apply(lambda x: adjuster.get_segments(x))
    helix_ranges = segment_info.apply(lambda x: ';'.join([':'.join(list(map(str, y))) for y in x[2]]))
    helix_crick = segment_info.apply(lambda x: x[0])
    data['detected_helix_ranges'] = helix_ranges
    data = data.drop(['seq', 'seq_pred', 'linkers'], axis='columns')

    def get_helices(row, ms='sequence'):
        if not row['detected_helix_ranges']:
            return []
        ranges = row['detected_helix_ranges'].split(';')
        rr = []
        for rg in ranges:
            l, r = map(int, rg.split(':'))
            rr.append(row[ms][l:r])
        return rr

    data['detected_helices'] = data.apply(lambda x: get_helices(x), axis=1)
    data['detected_helices_crick'] = helix_crick
    data['predicted_rotation'] = data['detected_helices_crick'].apply(lambda x: adjuster.get_rotation(*x))
    if config.get('is_test', False):
        return tester.get_metrics(test_data, sequences)
    return (data,) + results[1:]
