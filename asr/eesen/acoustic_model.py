import time

import torch

from ..nn_model import NNModel
from ..utils import ProgressTable


class EESENAcousticModel(NNModel):

    @staticmethod
    def load(path):
        data = torch.load(path)
        model = EESENAcousticModel(**data['meta'])
        model.load_state_dict(data['state_dict'])
        return model

    def __init__(self, input_size, hidden_size, num_layers, num_labels,
                 bidirectional=True, blank=0):
        super(EESENAcousticModel, self).__init__()
        self.optimizer = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional is True else 1
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            bidirectional=bidirectional)
        self.linear = torch.nn.Linear(
            hidden_size * num_directions, num_labels)
        self.activation = torch.nn.LogSoftmax(dim=2)
        self.blank = blank
        self.ctc_loss = torch.nn.CTCLoss(blank=blank)

    def init_weights(self):
        initrange = 0.1
        self.lstm.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, data):
        output, _ = self.lstm(data)
        output = self.linear(output)
        return self.activation(output)

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
            output = self.forward(data)
            return output.argmax(axis=2)

    def calc_loss(self, data, labels, input_lengths, label_lengths, device):
        data = data.detach().requires_grad_().to(device)
        labels = labels.to(device)
        input_lengths = input_lengths.to(device)
        label_lengths = label_lengths.to(device)
        output = self.forward(data)
        return self.ctc_loss(
            output, labels, input_lengths, label_lengths)

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
        progress_table = ProgressTable(
            'epoch', 'elapsed (sec)', 'tr loss',
            *['dev{} loss'.format(i+1) for i in range(len(dataloaders_dev))])
        progress_table.print_header()
        device = next(self.parameters()).device
        for epoch in range(epochs):
            start_time = time.time()
            loss_tr = 0
            for data, labels, input_lengths, label_lengths in dataloader_tr:
                self.optimizer.zero_grad()
                loss = self.calc_loss(
                    data.detach().requires_grad_(),
                    labels, input_lengths, label_lengths, device)
                loss.backward()
                self.optimizer.step()
                loss_tr += loss.item()
            losses_dev = []
            for dataloader_dev in dataloaders_dev:
                loss_dev = 0
                for data, labels, input_lengths, label_lengths in dataloader_dev:
                    loss = self.calc_loss(
                        data, labels, input_lengths, label_lengths, device)
                    loss_dev += loss.item()
                losses_dev.append(loss_dev)
            progress_table.print_row(
                epoch, time.time() - start_time, loss_tr, *losses_dev)

    def save(self, path):
        data = {
            'meta': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'num_labels': self.num_labels,
                'bidirectional': self.bidirectional,
                'blank': self.blank
            },
            'state_dict': self.state_dict()
        }
        torch.save(data, path)
