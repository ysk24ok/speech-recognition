import argparse
import os

import torch
from torch.utils.data import DataLoader

from asr.eesen import EESENAcousticModel
from asr import phonemes
from asr.dataset import (
    AudioDataset,
    IterableAudioDataset,
    TrainingDatasetRepository,
    DevelopmentDatasetRepository
)
from asr.feature import FeatureParams
from asr.label_table import PhonemeTable
from asr.utils import collate_for_ctc


parser = argparse.ArgumentParser()
parser.add_argument('workdir', type=str,
                    help='Directory path where files are loaded and saved')
parser.add_argument('--feature-params-file', type=str,
                    default='feature_params.json',
                    help='Feature params file name')
parser.add_argument('--training-data-dirname', type=str, default='trdir',
                    help='Directory name where training data are saved')
parser.add_argument('--development-data-dirname', type=str, default='devdir',
                    help='Directory name where development data are saved')
parser.add_argument('--hidden-size', type=int, default=256,
                    help='hidden layer size of an acoustic model')
parser.add_argument('--num-layers', type=int, default=4,
                    help='number of layers of an acoustic model')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='Optimizer to use')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs for training')
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='Device string')
parser.add_argument('--model-file', type=str, default="model.bin",
                    help='Model file to save')
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()

phoneme_table = PhonemeTable()
phoneme_table.add_labels(phonemes)
print('Loading training data ...')
training_data_dirpath = os.path.join(args.workdir, args.training_data_dirname)
repository_tr = TrainingDatasetRepository(training_data_dirpath)
dataset_tr = IterableAudioDataset(repository_tr, phoneme_table)
dataloader_tr = DataLoader(
    dataset_tr, batch_size=args.batch_size, collate_fn=collate_for_ctc)
print('Loading development data ...')
development_data_dirpath = os.path.join(args.workdir,
                                        args.development_data_dirname)
repository_dev = DevelopmentDatasetRepository(development_data_dirpath)
dataloaders_dev = []
for dataset_dev in AudioDataset.load_all(repository_dev, phoneme_table):
    dataloader_dev = DataLoader(
        dataset_dev, batch_size=args.batch_size, collate_fn=collate_for_ctc)
    dataloaders_dev.append(dataloader_dev)

feature_params_path = os.path.join(args.workdir, args.feature_params_file)
feature_params = FeatureParams.load(feature_params_path)

model_path = os.path.join(args.workdir, args.model_file)
if args.resume is True:
    print('Loading model ...')
    model = EESENAcousticModel.load(model_path)
else:
    print('Initializing model ...')
    blank = phoneme_table.get_blank_id()
    model = EESENAcousticModel(
        feature_params.feature_size, args.hidden_size, args.num_layers,
        phoneme_table.num_labels(), blank=blank)
model.to(torch.device(args.device))
model.set_optimizer(args.optimizer, args.lr)
print('Training ...')
model.train(dataloader_tr, dataloaders_dev, args.epochs)
print('Saving model ...')
model.save(model_path)
