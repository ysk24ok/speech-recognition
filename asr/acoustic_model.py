import time
from abc import ABCMeta, abstractmethod

import torch

from asr.utils import ProgressTable


class AcousticModel(metaclass=ABCMeta):

    def set_optimizer(self, optimizer_type, lr):
        """Set optimizer
        Args:
            optimizer_type (str): optimizer type
            lr (float): learning rate
        """
        if optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.module.parameters(), lr=lr)
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.module.parameters(), lr=lr)
        else:
            raise ValueError('Optimizer type: {} is not supported.'.format(
                optimizer_type))

    @abstractmethod
    def train(self):
        """Train the acoustic model."""
        pass

    @abstractmethod
    def save(self, path):
        """Save model.
        Args:
            path (str): Path to save
        """
        pass

    @staticmethod
    def load(path):
        """Load model
        Args:
            path (str): Path to load
        Returns:
            instance of AcousticModel
        """
        data = torch.load(path)
        if data['model_type'] == 'eesen':
            model = EESENAcousticModel(**data['meta'])
        else:
            raise ValueError('model_type: {} is not supported.')
        model.module.load_state_dict(data['module']['state_dict'])
        return model


class EESENModule(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_labels,
                 bidirectional=True):
        super(EESENModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.num_directions = 2 if bidirectional is True else 1
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            bidirectional=bidirectional)
        self.linear = torch.nn.Linear(
            hidden_size * self.num_directions, num_labels)
        self.activation = torch.nn.LogSoftmax(dim=2)

    def init_weights(self):
        initrange = 0.1
        self.lstm.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, data):
        output, _ = self.lstm(data)
        output = self.linear(output)
        return self.activation(output)


class EESENAcousticModel(AcousticModel):

    def __init__(self, input_size, hidden_size, num_layers, num_labels,
                 device=torch.device('cpu')):
        self.module = EESENModule(
            input_size, hidden_size, num_layers, num_labels).to(device)
        self.ctc_loss = torch.nn.CTCLoss().to(device)
        self.optimizer = None
        self.optimizer_type = None
        self.device = device

    def to(self, device):
        self.module = self.module.to(device)
        self.device = device

    def predict(self, data):
        """
        Args:
            data (torch.Tensor):
                shape=(sequence length, batch size, feature size)
        Returns:
            torch.Tensor: tensor whose shape is (sequence length, batch size)
                and each element is a label index
        """
        with torch.no_grad():
            output = self.module(data)
            return output.argmax(axis=2)

    def train(self, dataloader_tr, dataloaders_dev, epochs):
        """
        Args:
            dataloader_tr (torch.utils.data.dataloader.DataLoader):
                Dataloader object to be used for training
            dataloaders_dev (list[torch.utils.data.dataloader.DataLoader]):
                Dataloader objects to be used for development
            epochs (int): number of epochs
        """
        if self.optimizer is None:
            msg = 'Optimizer is not set. Call set_optimizer method first.'
            raise ValueError(msg)
        start_time = time.time()
        progress_table = ProgressTable(
            'epoch', 'elapsed (sec)', 'tr loss',
            *['dev{} loss'.format(i+1) for i in range(len(dataloaders_dev))])
        progress_table.print_header()
        for epoch in range(epochs):
            loss_tr = 0
            for data, labels, input_lengths, label_lengths in dataloader_tr:
                self.optimizer.zero_grad()
                data = data.detach().requires_grad_().to(self.device)
                labels = labels.to(self.device)
                input_lengths = input_lengths.to(self.device)
                label_lengths = label_lengths.to(self.device)
                output = self.module(data)
                loss = self.ctc_loss(
                    output, labels, input_lengths, label_lengths)
                loss.backward()
                self.optimizer.step()
                loss_tr += loss.item()
            losses_dev = []
            for dataloader_dev in dataloaders_dev:
                loss_dev = 0
                for data, labels, input_lengths, label_lengths in dataloader_dev:
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    input_lengths = input_lengths.to(self.device)
                    label_lengths = label_lengths.to(self.device)
                    output = self.module(data)
                    loss = self.ctc_loss(
                        output, labels, input_lengths, label_lengths)
                    loss_dev += loss.item()
                losses_dev.append(loss_dev)
            progress_table.print_row(
                epoch, time.time() - start_time, loss_tr, *losses_dev)
            start_time = time.time()

    def save(self, path):
        data = {
            'model_type': 'eesen',
            'meta': {
                'input_size': self.module.input_size,
                'hidden_size': self.module.hidden_size,
                'num_layers': self.module.num_layers,
                'num_labels': self.module.num_labels
            },
            'module': {
                'state_dict': self.module.state_dict()
            }
        }
        torch.save(data, path)
