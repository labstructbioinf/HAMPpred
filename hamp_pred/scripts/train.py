import argparse

from hamp_pred.src.predictor import Predictor
from hamp_pred.src.predictor_config import DEFAULT_CONF


def main(args):
    w = Predictor(args.model_name, DEFAULT_CONF)
    return w.train(args.training_data)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='hamp_crick_ensemble',
                        help='Name of model to train')
    parser.add_argument('--model_type', default='base_linear',
                        help='Type of model to use, like linear, cnn, resnet etc.')
    parser.add_argument('--training_data', default='data/input/dataset_MA_crdev.tsv')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run()
