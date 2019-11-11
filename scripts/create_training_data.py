import argparse

from asr import phonemes, kana_phoneme_mapping
from asr.dataset import CSJParser, AudioDataset
from asr.decoder import Lexicon
from asr.feature import FeatureParams


parser = argparse.ArgumentParser()
parser.add_argument('--fft-size', type=int, default=512, help='FFT size')
parser.add_argument('--frame-length', type=float, default=0.025,
                    help='frame length in seconds')
parser.add_argument('--hop-length', type=float, default=0.01,
                    help='frame interal in seconds')
parser.add_argument('--feature-size', type=int, default=80,
                    help='Feature size')
parser.add_argument('--feature-params-path', type=str,
                    default='feature_params.json',
                    help='Feature params path to save')
parser.add_argument('--training-data-path', type=str,
                    default='training_data.bin',
                    help='Training data path to save')
parser.add_argument('--dataset-type', type=str, default='csj',
                    help='Dataset type to use')
parser.add_argument('path', type=str, help='Dataset path')
args = parser.parse_args()

print('Creating training data ...')
feature_params = FeatureParams(
    fft_size=args.fft_size,
    frame_length=args.frame_length,
    hop_length=args.hop_length,
    feature_size=args.feature_size
)
lexicon = Lexicon(phonemes, kana_phoneme_mapping)
if args.dataset_type == 'csj':
    csj_parser = CSJParser(args.path, lexicon)
    data, labels = csj_parser.parse(feature_params)
else:
    raise ValueError('dataset_type: {} is not supported'.format(
        args.dataset_type))
dataset = AudioDataset(data, labels)
print('Saving trainig data ...')
feature_params.save(args.feature_params_path)
dataset.save(args.training_data_path)
