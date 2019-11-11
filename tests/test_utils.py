import torch
from torch.utils.data import DataLoader, Dataset

from asr import utils


def test_pad_labels():
    data = [
        torch.Tensor(5,),
        torch.Tensor(7,),
        torch.Tensor(3,)
    ]
    got = utils.pad_labels(data)
    assert(got.shape == (3, 7))


def test_pad_sequence():
    feature_size = 8
    data = [
        torch.Tensor(5, feature_size),
        torch.Tensor(8, feature_size),
        torch.Tensor(3, feature_size)
    ]
    got = utils.pad_sequence(data)
    assert(got.shape == (8, 3, feature_size))


class DatasetForTest(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


def test_collate_for_ctc():
    feature_size = 8
    batch_size = 2
    data = [
        torch.Tensor(5, feature_size),
        torch.Tensor(8, feature_size),
        torch.Tensor(3, feature_size),
        torch.Tensor(6, feature_size)
    ]
    labels = [
        torch.Tensor(4,),
        torch.Tensor(7,),
        torch.Tensor(3,),
        torch.Tensor(4,)
    ]
    dataset = DatasetForTest(data, labels)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=utils.collate_for_ctc)
    iterator = iter(dataloader)
    inputs, labels, input_lengths, label_lengths = iterator.__next__()
    assert(isinstance(inputs, torch.Tensor))
    assert(inputs.shape == (8, batch_size, feature_size))
    assert(isinstance(labels, torch.Tensor))
    assert(labels.shape == (batch_size, 7))
    assert(isinstance(input_lengths, torch.Tensor))
    assert(input_lengths.tolist() == [5, 8])
    assert(isinstance(label_lengths, torch.Tensor))
    assert(label_lengths.tolist() == [4, 7])
    inputs, labels, input_lengths, label_lengths = iterator.__next__()
    assert(isinstance(inputs, torch.Tensor))
    assert(inputs.shape == (6, batch_size, feature_size))
    assert(isinstance(labels, torch.Tensor))
    assert(labels.shape == (batch_size, 4))
    assert(isinstance(input_lengths, torch.Tensor))
    assert(input_lengths.tolist() == [3, 6])
    assert(isinstance(label_lengths, torch.Tensor))
    assert(label_lengths.tolist() == [3, 4])
