import time

import torch
from torch.nn.utils.rnn import pad_sequence

from .listener import LSTMListener, PyramidalLSTMListener
from .speller import LSTMSpeller, AttentionSpeller
from ..nn_model import NNModel
from ..utils import ProgressTable


class ListenAttendSpell(NNModel):

    @staticmethod
    def load(path):
        data = torch.load(path)
        model = ListenAttendSpell(**data['meta'])
        model.load_state_dict(data['state_dict'])
        return model

    @staticmethod
    def collate(sample, pad_id=0, bos_id=2, eos_id=3, reverse_input=True):
        # TODO: use functools.partial to pass arguments
        inputs = []
        labels_input = []
        labels_output = []
        for input, label in sample:
            inputs.append(input)
            label_input = torch.cat((torch.Tensor([bos_id]).long(), label))
            labels_input.append(label_input)
            label_output = torch.cat((label, torch.Tensor([eos_id]).long()))
            labels_output.append(label_output)
        padded_input = pad_sequence(inputs, padding_value=pad_id)
        if reverse_input is True:
            seq_size = padded_input.shape[0]
            padded_input = padded_input[torch.arange(seq_size-1, -1, -1), :, :]
        return (
            padded_input,
            pad_sequence(labels_input, padding_value=pad_id).T,
            pad_sequence(labels_output, padding_value=pad_id).T
        )

    def __init__(self, input_size, hidden_size, vocab_size, embedding_size,
                 listener_type='pyramidal_lstm',
                 listener_num_layers=2,
                 listener_bidirectional=True,
                 speller_type='attention_lstm',
                 pad_id=0):
        super(ListenAttendSpell, self).__init__()
        self.optimizer = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.listener_type = listener_type
        self.listener_num_layers = listener_num_layers
        self.listener_bidirectional = listener_bidirectional
        self.speller_type = speller_type
        self.pad_id = pad_id
        self._set_listener(listener_type, input_size, hidden_size,
                           listener_num_layers, listener_bidirectional)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self._set_speller(speller_type, embedding_size,
                          hidden_size * self.listener.num_directions,
                          vocab_size)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    def _set_listener(self, listener_type, input_size, hidden_size, num_layers,
                      bidirectional):
        if listener_type == 'lstm':
            self.listener = LSTMListener(input_size, hidden_size, num_layers,
                                         bidirectional=bidirectional)
        elif listener_type == 'pyramidal_lstm':
            self.listener = PyramidalLSTMListener(
                input_size, hidden_size, num_layers,
                bidirectional=bidirectional)
        else:
            raise ValueError("listener_type must be one of the following: "
                             "'lstm' or 'pyramidal_lstm'")

    def _set_speller(self, speller_type, input_size, hidden_size, vocab_size):
        # TODO: number of layers of speller is assumed as 2
        speller_num_layers = 2
        if speller_type == 'lstm':
            self.speller = LSTMSpeller(
                input_size, hidden_size, vocab_size, speller_num_layers)
        elif speller_type == 'attention_lstm':
            self.speller = AttentionSpeller(
                input_size, hidden_size, vocab_size, speller_num_layers)
        else:
            raise ValueError("speller_type must be one of the following: "
                             "'lstm' or 'attention_lstm'")

    def forward(self, inputs, labels=None):
        batch_size = inputs.shape[1]
        hidden = self.listener.init_hidden(batch_size)
        listener_outputs, listener_hidden = self.listener(inputs, hidden)
        decoder_inputs = torch.transpose(self.embedding(labels), 0, 1)
        # NOTE: reshape hidden state whose shape is
        #       (layer size x num_directions, batch size, hidden size)
        #       into
        #       (layer size, batch size, hidden size x num_directions)
        # TODO: number of layers of speller is assumed as 2
        if self.listener.num_directions == 2:
            h = listener_hidden[0]
            c = listener_hidden[1]
            speller_hidden = (
                torch.cat((
                    torch.cat((h[-4:-3], h[-3:-2]), dim=2),
                    torch.cat((h[-2:-1], h[-1:]), dim=2),
                ), dim=0),
                torch.cat((
                    torch.cat((c[-4:-3], c[-3:-2]), dim=2),
                    torch.cat((c[-2:-1], c[-1:]), dim=2),
                ), dim=0)
            )
        else:
            speller_hidden = listener_hidden
        return self.speller(decoder_inputs, speller_hidden, listener_outputs)

    def calc_loss(self, data, labels_input, labels_output, device):
        data = data.to(device=device)
        labels_input = labels_input.to(device=device)
        labels_output = labels_output.to(device=device)
        output = self.forward(data, labels_input)
        # TODO: correct?
        return self.loss(
            torch.transpose(output, 0, 1).contiguous().view(
                -1, self.vocab_size),
            labels_output.contiguous().view(-1)
        )

    def train(self, dataloader_tr, dataloaders_dev, epochs):
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
            for data, labels_input, labels_output in dataloader_tr:
                self.optimizer.zero_grad()
                loss = self.calc_loss(
                    data.detach().requires_grad_(),
                    labels_input, labels_output, device)
                loss.backward()
                self.optimizer.step()
                loss_tr += loss.item()
            losses_dev = []
            for dataloader_dev in dataloaders_dev:
                loss_dev = 0
                for data, labels_input, labels_output in dataloader_dev:
                    loss = self.calc_loss(data, labels_input, labels_output,
                                          device)
                    loss_dev += loss.item()
                losses_dev.append(loss_dev)
            progress_table.print_row(
                epoch, time.time() - start_time, loss_tr, *losses_dev)

    def save(self, path):
        data = {
            'meta': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'vocab_size': self.vocab_size,
                'embedding_size': self.embedding_size,
                'listener_type': self.listener_type,
                'listener_num_layers': self.listener_num_layers,
                'listener_bidirectional': self.listener_bidirectional,
                'speller_type': self.speller_type,
                'pad_id': self.pad_id,
            },
            'state_dict': self.state_dict()
        }
        torch.save(data, path)
