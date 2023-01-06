def run(sequences, config=None):
    prev = type(config.get('predictor'))
    curr = type(config.get('predictor'))
    curr = curr('hamp_rot')
    seqs = prev('hamp_crick_single_sequence').predict(sequences, with_model=True)
    sequences = seqs[0].apply(lambda x: ''.join(x['detected_helices']), axis=1).tolist()
    sequences = list(filter(lambda x: x, sequences))
    if not sequences:
        print('Did not find any helices')
        return seqs
    result = curr.predict(sequences, with_model=False)
    seqs[0]['predicted_rotation'] = result['predicted_rotation']
    return seqs
