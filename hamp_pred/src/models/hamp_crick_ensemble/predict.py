from hamp_pred.src.models.hamp_crick_ensemble.test import Tester


def run(sequences, config=None):
    prev = type(config.get('predictor'))
    curr = type(config.get('predictor'))
    curr = curr('hamp_rot')
    if config.get('is_test', False):
        tester = config.get('tester') or Tester(scale=1, out_column='predicted_rotation')
        test_data = sequences
        sequences = tester.get_squences(sequences)
    seqs = prev('hamp_crick_single_sequence').predict(sequences, with_model=True)
    sequences = seqs[0].apply(lambda x: ''.join(x['detected_helices']), axis=1).tolist()
    sequences = list(filter(lambda x: x, sequences))
    if not sequences:
        print('Did not find any helices')
        return seqs
    result = curr.predict(sequences, with_model=False)
    seqs[0]['predicted_rotation'] = result['predicted_rotation']
    if config.get('is_test', False):
        return tester.get_metrics(test_data, seqs[0])
    return seqs
