from abc import ABCMeta, abstractmethod

import torch


class NNModel(torch.nn.Module, metaclass=ABCMeta):

    def set_optimizer(self, optimizer_type, lr):
        """Set optimizer
        Args:
            optimizer_type (str): optimizer type
            lr (float): learning rate
        """
        if optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
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
