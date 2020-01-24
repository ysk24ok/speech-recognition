import argparse
import os

import torch
from torch.utils.data import DataLoader

from asr.dataset import (
    AudioDataset,
    IterableAudioDataset,
    TrainingDatasetRepository,
    DevelopmentDatasetRepository
)
from asr.feature import FeatureParams
from asr.label_table import VocabularyTable
from asr.las import ListenAttendSpell


parser = argparse.ArgumentParser()
parser.add_argument('workdir', type=str,
                    help='Directory path where files are loaded and saved')
parser.add_argument('--feature-params-file', type=str,
                    default='feature_params.json',
                    help='Feature params file name')
parser.add_argument('--vocabulary-table-file', type=str, default='vocab.txt',
                    help='Vocabulary table file name')
parser.add_argument('--min-word-frequency', type=int, default=1,
                    help='If word frequency in a vocabulary table is '
                    'lower than this value, the word is ignored '
                    'and recognized as unknown.')
parser.add_argument('--model-file', type=str, default='model.bin',
                    help='Model file name')
parser.add_argument('--training-data-dirname', type=str, default='trdir',
                    help='Directory name where training data are saved')
parser.add_argument('--development-data-dirname', type=str, default='devdir',
                    help='Directory name where development data are saved')
parser.add_argument('--hidden-size', type=int, default=128,
                    help='Hidden size of LAS network')
parser.add_argument('--embedding-size', type=int, default=64,
                    help='Embedding size for a word')
parser.add_argument('--listener-type', type=str, default='pyramidal_lstm',
                    choices=('lstm', 'pyramidal_lstm'),
                    help='Listener type')
parser.add_argument('--listener-num-layers', type=int, default=2,
                    help='Number of layers of Listener. '
                         "If 'pyramidal_lstm' is specified to"
                         "'--listener-type', "
                         'Listener consists of 1 bottom LSTM layer '
                         'and n-1 pLSTM layer on top of it.')
parser.add_argument('--speller-type', type=str, default='attention_lstm',
                    choices=('lstm', 'attention_lstm'),
                    help='Speller type')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='Optimizer to use')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--epochs', type=int, default=5,
                    help='Number of epochs for training')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--device', type=str, default='cpu', help='Device string')
parser.add_argument('--resume', action='store_true',
                    help="Training resumes from a model file "
                         "specified by '--model-file' option")
args = parser.parse_args()

print('Loading vocabulary table ...')
vocab_path = os.path.join(args.workdir, args.vocabulary_table_file)
vocab_table = VocabularyTable.load(vocab_path,
                                   min_freq=args.min_word_frequency)
print('Loading training data ...')
training_data_dirpath = os.path.join(args.workdir, args.training_data_dirname)
repository_tr = TrainingDatasetRepository(training_data_dirpath)
dataset_tr = IterableAudioDataset(repository_tr, vocab_table)
dataloader_tr = DataLoader(dataset_tr, batch_size=args.batch_size,
                           collate_fn=ListenAttendSpell.collate)
print('Loading development data ...')
development_data_dirpath = os.path.join(args.workdir,
                                        args.development_data_dirname)
repository_dev = DevelopmentDatasetRepository(development_data_dirpath)
dataloaders_dev = []
for dataset_dev in AudioDataset.load_all(repository_dev, vocab_table):
    dataloader_dev = DataLoader(dataset_dev, batch_size=args.batch_size,
                                collate_fn=ListenAttendSpell.collate)
    dataloaders_dev.append(dataloader_dev)

feature_params_path = os.path.join(args.workdir, args.feature_params_file)
feature_params = FeatureParams.load(feature_params_path)

model_path = os.path.join(args.workdir, args.model_file)
if args.resume is True:
    print('Loading model ...')
    model = ListenAttendSpell.load(model_path)
else:
    print('Initializing model ...')
    vocab_size = vocab_table.num_labels()
    pad_id = vocab_table.get_pad_id()
    model = ListenAttendSpell(
        feature_params.feature_size, args.hidden_size, vocab_size,
        args.embedding_size,
        listener_type=args.listener_type,
        listener_num_layers=args.listener_num_layers,
        listener_bidirectional=True,
        speller_type=args.speller_type,
        pad_id=pad_id)
model.to(torch.device(args.device))
model.set_optimizer(args.optimizer, args.lr)
print('Training ...')
model.train(dataloader_tr, dataloaders_dev, args.epochs)
print('Saving model ...')
model.save(model_path)
