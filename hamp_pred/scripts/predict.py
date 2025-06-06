import argparse
import pathlib
import pickle


def main(args):
    from hamp_pred.src.predictor import Predictor
    from hamp_pred.src.predictor_config import DEFAULT_CONF
    w = Predictor(args.model_name, config=DEFAULT_CONF)
    if args.test_sequences:
        result = w.predict(args.test_sequences)
    else:
        result = w.process_data(args.test_data, kind=args.data_kind, path=args.test_data)
    if args.save_path:
        if args.results_format == 'pickle':
            with open(args.save_path, 'wb') as dp:
                pickle.dump(result, dp)
        else:
            result.to_csv(args.save_path, index=False, sep='\t')
    return result


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='hamp_crick_ensemble',
                        help='Name of model to train',
                        choices=['hamp_crick_ensemble', 'hamp_rot', 'hamp_crick_single_sequence'])
    parser.add_argument('--model_type', default='base_linear',
                        help='Type of model to use, like linear, cnn, resnet etc.')
    parser.add_argument('--test_sequences', default='', nargs='+')
    parser.add_argument('--test_data', default=pathlib.Path(__file__).parent.parent.parent.joinpath(
        'data/input/example_hamp_seq.fasta'))
    parser.add_argument('--data_kind', default='fasta',
                        help='Kind of data to process',
                        choices=['msa', 'fasta'])
    parser.add_argument('--save_path', help='Path to save results')
    parser.add_argument('--results_format', help='Format of results (tsv or pickle)',
                        default='tsv', choices=['tsv', 'pickle'])
    args = parser.parse_args()
    return main(args)


if __name__ == '__main__':
    print(run())
