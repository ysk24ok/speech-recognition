import argparse

from asr import phonemes, kana2phonemes
from asr.acoustic_labels import MonophoneLabels
from asr.dataset import (
    TrainingDatasetRepository,
    DevelopmentDatasetRepository
)
from asr.dataset.csj import CSJParser
from asr.decoder import Corpus, Lexicon
from asr.feature import FeatureParams


def extract_feature(csj_parser, talk_sets, repository, feature_params):
    if isinstance(repository, TrainingDatasetRepository):
        tr_or_dev = 'training'
    else:
        tr_or_dev = 'development'
    for i, talks in enumerate(talk_sets):
        print('Extracting features of {}/{} of {} data'.format(
            i + 1, len(talk_sets), tr_or_dev))
        data, targets = csj_parser.parse(talks, feature_params)
        print('Saving {}/{} of {} data ...'.format(
            i + 1, len(talk_sets), tr_or_dev))
        repository.save(data, targets)


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
parser.add_argument('--training-data-dirpath', type=str, default='trdir',
                    help='Directory path where training data are saved')
parser.add_argument('--development-data-dirpath', type=str, default='devdir',
                    help='Directory path where development data are saved')
parser.add_argument('--corpus-path', type=str, default='corpus.txt',
                    help='File path where corpus is saved')
parser.add_argument('--lexicon-path', type=str, default='lexicon.txt',
                    help='File path where lexicon is saved')
parser.add_argument('--training-data-file-count', type=int,
                    default=4, help='Training data file count to save')
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
repository_tr = TrainingDatasetRepository(args.training_data_dirpath)
repository_dev = DevelopmentDatasetRepository(args.development_data_dirpath)
corpus = Corpus()
acoustic_labels = MonophoneLabels(phonemes, kana2phonemes)
lexicon = Lexicon(acoustic_labels)
if args.dataset_type == 'csj':
    csj_parser = CSJParser(args.path, corpus, lexicon)
    talk_sets_tr, talk_sets_dev = csj_parser.get_talks(
        args.training_data_file_count)
    extract_feature(csj_parser, talk_sets_tr, repository_tr, feature_params)
    extract_feature(csj_parser, talk_sets_dev, repository_dev, feature_params)
else:
    raise ValueError('dataset_type: {} is not supported'.format(
        args.dataset_type))
feature_params.save(args.feature_params_path)
corpus.save(args.corpus_path)
lexicon.save(args.lexicon_path)
