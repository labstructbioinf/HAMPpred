import pathlib

from Bio import SeqIO

from external.SequenceEncoding.SequenceEncoding import SequenceEncoding, get_dict
from ma_predictor.src.input_prep.encode import OneHotEncoderSeq, RadianEncoder, MixedEncoder
from ma_predictor.src.input_prep.prepare_sequence import MultiChainOperator, SeqWindow
from ma_predictor.src.models.common.models import BaseConvolutionalWrapper, BaseLinearWrapper

class PredictionConfig:
    def __init__(self, model, operator,
                 model_config=None,
                 test_path=None):
        self.model = model
        self.operator = operator
        self.model_config = model_config or {}
        self.test_path = test_path
        self.task = None
        self.version = None
        self.test_ids = None
        self.val_ids = None
        self.ids = None

    def set_task(self, name, version):
        self.task = name
        self.version = version
        self.model_config['task'] = self.task
        self.model_config['version'] = self.version
    def set_val_ids(self, val_ids):
        self.val_ids = val_ids
    def set_ids(self, ids):
        self.ids = ids

    def set_test_ids(self):
        if self.test_path:
            ids = [seq.id for seq in SeqIO.parse(self.test_path, "fasta")]
            self.test_ids = ids

    def dump(self):
        results = {}
        for att in dir(self):
            if not att.startswith('__'):
                results[att] = getattr(self, att)
        return results


test_path = pathlib.Path(__file__).parent.parent.parent.joinpath('data/input/test_seq.fasta')
enc = MixedEncoder()
enc.set_types(SequenceEncoding.encoding_types)
operator = MultiChainOperator(OneHotEncoderSeq(), SeqWindow(19, 19), RadianEncoder(100),  SeqWindow(19, 19, null_char=[[0]]))
operator_exteranl = MultiChainOperator(enc, SeqWindow(19, 19), RadianEncoder(100),  SeqWindow(19, 19, null_char=[[0]]))

SEQ_ENCODING_EXTERNAL = PredictionConfig(BaseConvolutionalWrapper,
                                        operator_exteranl,
                                        {'activation': 'tanh'},
                                        test_path=test_path)

DEFAULT_CONF = PredictionConfig(BaseConvolutionalWrapper,
                                operator,
                                {'activation': 'tanh'},
                                test_path=test_path)