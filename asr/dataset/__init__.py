import glob
import os
import pickle
from abc import ABCMeta, abstractmethod

import torch


class DatasetParser(metaclass=ABCMeta):

    @abstractmethod
    def parse(self):
        pass


class AudioDataset(torch.utils.data.Dataset):

    """Subclass of torch.utils.data.Dataset.
    This class is suitable for data which is small enough to fit in memory.
    """

    def __init__(self, data, labels):
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert isinstance(self.data[idx], torch.Tensor)
        assert isinstance(self.labels[idx], torch.Tensor)
        return self.data[idx], self.labels[idx]

    @staticmethod
    def load_all(repository, label_table):
        """Load all files from the repository as list of AudioDataset objects.
        The length of the list is equal to the number of files saved on disk.

        Args:
            repository (asr.dataset.DatasetRepository):
                DatasetRepository object
            label_table (asr.label_table.LabelTable): LabelTable object
        Returns:
            list[asr.dataset.AudioDataset]: AudioDataset objects
        """
        return [AudioDataset(data, labels)
                for data, labels in repository.load_next(label_table)]

    @staticmethod
    def load_concat(repository, label_table):
        """Load all files from the repository into one AudioDataset object.

        Args:
            repository (asr.dataset.DatasetRepository):
                DatasetRepository object
            label_table (asr.label_table.LabelTable): LabelTable object
        Returns:
            asr.dataset.AudioDataset: AudioDataset object
        """
        data, labels = [], []
        for d, label in repository.load_next(label_table):
            data.extend(d)
            labels.extend(label)
        return AudioDataset(data, labels)


class IterableAudioDataset(torch.utils.data.IterableDataset):

    """Subclass of torch.utils.data.IterableDataset.
    This class is suitable for data which is so big that cannot fit in memory.
    """

    def __init__(self, repository, label_table):
        self.repository = repository
        self.label_table = label_table

    def __iter__(self):
        for data, labels in self.repository.load_next(self.label_table):
            for d, label in zip(data, labels):
                yield d, label


def _generate_idx(start=0):
    idx = start
    while True:
        yield idx
        idx += 1


class DatasetRepository(object):

    def __init__(self, dirpath):
        self.dirpath = dirpath
        self._idx_generator = _generate_idx()

    def save(self, data, labels):
        """Save data and labels as one file with a specified prefix.

        Args:
            data (list[numpy.ndarray]):
                list of array whose shape is (number of frames, feature size)
            labels (list[list[str]]):
                list of array whose shape is (number of labels,)
        """
        # NOTE: Save data and labels as numpy.ndarray
        #       because saving them as torch.Tensor requires a lot of memory
        filename_suffix = next(self._idx_generator)
        fname = '{}{:0>3}.pkl'.format(self.filename_prefix, filename_suffix)
        fpath = os.path.join(self.dirpath, fname)
        with open(fpath, 'wb') as f:
            obj = {'data': data, 'labels': labels}
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_next(self, label_table):
        """Load one file with a specified prefix as torch.Tensor

        Returns:
            generator: generator object
        """
        for fpath in self._get_filepaths():
            yield self._load_one(fpath, label_table)

    def _get_filepaths(self):
        fname = '{}*.pkl'.format(self.filename_prefix)
        return glob.glob(os.path.join(self.dirpath, fname))

    def _load_one(self, filepath, label_table):
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return self._to_torch(
            obj['data'],
            self._convert_label_to_id(obj['labels'], label_table)
        )

    def _convert_label_to_id(self, list_of_labels, label_table):
        list_of_label_ids = []
        for labels in list_of_labels:
            label_ids = []
            for label in labels:
                label_ids.append(label_table.get_label_id(label))
            list_of_label_ids.append(label_ids)
        return list_of_label_ids

    def _to_torch(self, data, labels):
        return (
            [torch.from_numpy(d) for d in data],
            [torch.Tensor(l).long() for l in labels]
        )


class TrainingDatasetRepository(DatasetRepository):

    def __init__(self, dirpath):
        super(TrainingDatasetRepository, self).__init__(dirpath)
        self.filename_prefix = 'tr'


class DevelopmentDatasetRepository(DatasetRepository):

    def __init__(self, dirpath):
        super(DevelopmentDatasetRepository, self).__init__(dirpath)
        self.filename_prefix = 'dev'
