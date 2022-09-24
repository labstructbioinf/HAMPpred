import argparse
import pickle

from ma_predictor.src.predictor import Predictor
from ma_predictor.src.predictor_config import DEFAULT_CONF


def main(args):
    w = Predictor(args.model_name, DEFAULT_CONF)
    result = w.process_data(args.test_data, kind=args.data_kind)
    if args.save_path:
        with open(args.save_path, 'wb') as dp:
            pickle.dump(result, dp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='two_helix_crdev',
                        help='Name of model to train')
    parser.add_argument('--model_type', default='base_linear',
                        help='Type of model to use, like linear, cnn, resnet etc.')
    parser.add_argument('--test_data', default='data/input/ma_alignment.aln')
    parser.add_argument('--data_kind', default='msa',
                        help='Kind of data to process',
                        choices=['msa', 'fasta'])
    parser.add_argument('--save_path', help='Path to save results')
    parser.add_argument('--results_format', help='Format of results',
                        default='pickle')
    args = parser.parse_args()
    main(args)
