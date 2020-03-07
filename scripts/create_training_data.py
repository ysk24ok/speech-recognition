import argparse
import os

from asr.dataset import (
    TrainingDatasetRepository,
    DevelopmentDatasetRepository
)
from asr.dataset.csj import CSJParser
from asr.eesen.decoder import Lexicon
from asr.feature import FeatureParams


def create_repository(workdir, dirname, RepositoryClass):
    dirpath = os.path.join(workdir, dirname)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    return RepositoryClass(dirpath)


def extract_feature(csj_parser, talk_sets, repository, feature_params,
                    label_type, use_subset):
    if isinstance(repository, TrainingDatasetRepository):
        tr_or_dev = 'training'
    else:
        tr_or_dev = 'development'
    for i, talks in enumerate(talk_sets):
        print('Extracting features of {}/{} of {} data'.format(
            i + 1, len(talk_sets), tr_or_dev))
        data, targets = csj_parser.parse(talks, feature_params,
                                         label_type=label_type,
                                         only_core=use_subset)
        print('Saving {}/{} of {} data ...'.format(
            i + 1, len(talk_sets), tr_or_dev))
        repository.save(data, targets)


parser = argparse.ArgumentParser()
parser.add_argument('workdir', type=str,
                    help='Directory path where files are loaded and saved')
parser.add_argument('dataset_path', type=str, help='Dataset path')
parser.add_argument('--fft-size', type=int, default=512, help='FFT size')
parser.add_argument('--frame-length', type=float, default=0.025,
                    help='frame length in seconds')
parser.add_argument('--hop-length', type=float, default=0.01,
                    help='frame interal in seconds')
parser.add_argument('--feature-size', type=int, default=80,
                    help='Feature size')
parser.add_argument('--feature-params-file', type=str,
                    default='feature_params.json',
                    help='Feature params file name')
parser.add_argument('--label-type', type=str, default='word',
                    choices=('phoneme', 'word'),
                    help='Targets to train.')
parser.add_argument('--vocabulary-table-file', type=str, default='vocab.txt',
                    help='Vocabulary table file name. '
                    "If 'word' is not specified to '--label-type', "
                    "this option is ignored.")
parser.add_argument('--min-word-frequency', type=int, default=1,
                    help='If word frequency in a vocabulary table is '
                    'lower than this value, the word is ignored '
                    'and recognized as unknown. '
                    "If 'word' is not specified to '--label-type', "
                    "this option is ignored.")
parser.add_argument('--training-data-dirname', type=str, default='trdir',
                    help='Directory name where training data are saved')
parser.add_argument('--development-data-dirname', type=str, default='devdir',
                    help='Directory name where development data are saved')
parser.add_argument('--create-lexicon', action='store_true',
                    help="A flag whether to create lexicon. "
                    "If 'word' is specified to '--label-type', "
                    "this option is ignored.")
parser.add_argument('--lexicon-file', type=str, default='lexicon.txt',
                    help='Lexicon file name. '
                    "If 'word' is specified to '--label-type', "
                    "this option is ignored.")
parser.add_argument('--training-data-file-count', type=int,
                    default=4, help='Training data file count to save')
parser.add_argument('--dataset-type', type=str, default='csj',
                    help='Dataset type to use')
parser.add_argument('--use-subset', action='store_true',
                    help='A flag whether to use a subset of the dataset')
args = parser.parse_args()

feature_params = FeatureParams(
    fft_size=args.fft_size,
    frame_length=args.frame_length,
    hop_length=args.hop_length,
    feature_size=args.feature_size
)
repository_tr = create_repository(args.workdir, args.training_data_dirname,
                                  TrainingDatasetRepository)
repository_dev = create_repository(args.workdir, args.development_data_dirname,
                                   DevelopmentDatasetRepository)
print('Creating training data ...')
if args.dataset_type == 'csj':
    if args.create_lexicon and args.label_type != 'word':
        lexicon = Lexicon()
        csj_parser = CSJParser(args.dataset_path, lexicon=lexicon)
    else:
        csj_parser = CSJParser(args.dataset_path)
    talk_sets_tr, talk_sets_dev = csj_parser.get_talks(
        args.training_data_file_count)
    extract_feature(csj_parser, talk_sets_tr, repository_tr, feature_params,
                    args.label_type, args.use_subset)
    extract_feature(csj_parser, talk_sets_dev, repository_dev, feature_params,
                    args.label_type, args.use_subset)
    if args.create_lexicon and args.label_type != 'word':
        print('Saving lexicon ...')
        lexicon_path = os.path.join(args.workdir, args.lexicon_file)
        lexicon.save(lexicon_path)
else:
    raise ValueError('dataset_type: {} is not supported'.format(
        args.dataset_type))
print('Saving feature params ...')
feature_params_path = os.path.join(args.workdir, args.feature_params_file)
feature_params.save(feature_params_path)
