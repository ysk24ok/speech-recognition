import argparse

import torch
from torch.utils.data import DataLoader

from asr import acoustic_model
from asr import phonemes
from asr.dataset import (
    TrainingDatasetRepository,
    DevelopmentDatasetRepository
)
from asr.feature import FeatureParams
from asr.utils import collate_for_ctc


parser = argparse.ArgumentParser()
parser.add_argument('--model-type', type=str, default='eesen',
                    help='Model type to train')
parser.add_argument('--feature-params-path', type=str,
                    default='feature_params.json',
                    help='Feature params path to load')
parser.add_argument('--training-data-path', type=str,
                    default='training_data.bin',
                    help='Training data path to load')
parser.add_argument('--development-data-path', type=str,
                    default='development_data.bin',
                    help='Development data path to load')
parser.add_argument('--hidden-size', type=int, default=256,
                    help='hidden layer size of an acoustic model')
parser.add_argument('--num-layers', type=int, default=4,
                    help='number of layers of an acoustic model')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='Optimizer to use')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs for training')
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='Device string')
parser.add_argument('--model-path', type=str, default="model.bin",
                    help='Model path to save')
args = parser.parse_args()

print('Loading training data ...')
dataset_tr = TrainingDatasetRepository(args.training_data_path).load()
dataset_tr.to_torch()
dataloader_tr = DataLoader(dataset_tr, batch_size=args.batch_size,
                           collate_fn=collate_for_ctc)
print('Loading development data ...')
dataloaders_dev = []
datasets_dev = DevelopmentDatasetRepository(args.development_data_path).load()
for dataset_dev in datasets_dev:
    dataset_dev.to_torch()
    dataloader_dev = DataLoader(dataset_dev, batch_size=args.batch_size,
                                collate_fn=collate_for_ctc)
    dataloaders_dev.append(dataloader_dev)
print('Training ...')
num_labels = len(phonemes)
feature_params = FeatureParams.load(args.feature_params_path)
device = torch.device(args.device)
if args.model_type == 'eesen':
    model = acoustic_model.EESENAcousticModel(
        feature_params.feature_size, args.hidden_size, args.num_layers,
        num_labels, device)
else:
    raise ValueError('model_type: {} is not supported.'.format(
        args.model_type))
model.set_optimizer(args.optimizer, args.lr)
model.train(dataloader_tr, dataloaders_dev, args.epochs)
print('Saving model ...')
model.save(args.model_path)
