import numpy as np
import pandas as pd

from hamp_pred.src.input_prep.encode import OneHotEncoderSeq, RadianEncoder
from hamp_pred.src.input_prep.prepare_sequence import MultiChainOperator, SeqWindow
from hamp_pred.src.models.common.models import BaseLinearWrapper
from hamp_pred.src.models.hamp_rot.test import Tester


def run(sequences, config=None):
    prev = type(config.get('predictor'))
    curr = type(config.get('predictor'))
    curr = curr('hamp_rot', config=config.get('predictor').config)
    seqs = prev('hamp_crick_single_sequence').predict(sequences)
    sequences = seqs.apply(lambda x: ''.join(x['detected_helices']), axis=1).tolist()
    sequences = list(filter(lambda x: x, sequences))
    if not sequences:
        return None, None
    return curr.predict(sequences, with_model=True)
