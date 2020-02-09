import subprocess
from collections import defaultdict


class LabelTable(object):

    def __init__(self):
        self._labels = []
        self._label2id = {}

    def get_epsilon_id(self):
        return self.get_label_id('<epsilon>')

    def add_label(self, label):
        if label in self._label2id:
            return
        self._labels.append(label)
        self._label2id[label] = len(self._labels) - 1

    def add_labels(self, labels):
        for label in labels:
            self.add_label(label)

    def num_labels(self):
        return len(self._labels)

    def get_label_id(self, label):
        return self._label2id[label]

    def get_label(self, label_id):
        return self._labels[label_id]


class PhonemeTable(LabelTable):

    def __init__(self):
        super(PhonemeTable, self).__init__()
        self.add_label('<epsilon>')
        self.add_label('<blank>')
        self._auxiliary_labels = []

    def get_blank_id(self):
        return self.get_label_id('<blank>')

    def get_auxiliary_label_id(self, label):
        offset = len(self._labels)
        return self._auxiliary_labels.index(label) + offset

    def set_auxiliary_label(self, label):
        if label not in self._auxiliary_labels:
            self._auxiliary_labels.append(label)

    def get_all_labels(self):
        return {idx: label for idx, label in enumerate(self._labels)}

    def get_all_auxiliary_labels(self):
        offset = len(self._labels)
        return {idx + offset: label
                for idx, label in enumerate(self._auxiliary_labels)}


class VocabularyTable(LabelTable):

    def __init__(self, min_freq=1):
        super(VocabularyTable, self).__init__()
        self._basic_labels = ['<pad>', '<unk>', '<bos>', '<eos>']
        for label in self._basic_labels:
            super(VocabularyTable, self).add_label(label)
        self._freq_per_label = defaultdict(int)
        self._min_freq = min_freq

    def get_pad_id(self):
        return self.get_label_id('<pad>')

    def get_unk_id(self):
        return self.get_label_id('<unk>')

    def get_bos_id(self):
        return self.get_label_id('<bos>')

    def get_eos_id(self):
        return self.get_label_id('<eos>')

    def add_label(self, label):
        self._freq_per_label[label] += 1
        if self._freq_per_label[label] >= self._min_freq:
            super(VocabularyTable, self).add_label(label)

    def get_label_id(self, label):
        try:
            return self._label2id[label]
        except KeyError:
            return self.get_unk_id()

    def save(self, path):
        """Save vocabulary table to disk
        Args:
            path (str): vocabulary table filepath
        """
        with open(path, 'w') as f:
            for label in self._labels:
                if label in self._basic_labels:
                    continue
                f.write('{}\t{}\n'.format(label, self._freq_per_label[label]))
            for label, freq in self._freq_per_label.items():
                if label in self._label2id:
                    continue
                f.write('{}\t{}\n'.format(label, freq))

    @staticmethod
    def load(path, min_freq=1):
        """
        Args:
            path (str): vocabulary table filepath
            min_freq (int): minimum word frequency
        """
        with open(path) as f:
            vocabulary_table = VocabularyTable(min_freq=min_freq)
            for line in f:
                label, freq = line.split('\t')
                for _ in range(int(freq)):
                    vocabulary_table.add_label(label)
        return vocabulary_table


class VocabularySymbolTable(LabelTable):

    @staticmethod
    def create_symbol(path, corpus_path):
        """Create a vocabulary symbol table from a corpus
        Args:
            path (str): vocabulary symbol filepath
            corpus_path (str): filepath of a corpus
        """
        subprocess.run(['ngramsymbols', corpus_path, path])

    @staticmethod
    def load_symbol(path):
        """Read a vocabulary table from disk
        Args:
            path (str): vocabulary symbol filepath
        """
        vocab_symbol_table = VocabularySymbolTable()
        with open(path) as f:
            for l in f:
                word, word_id = l.split('\t')
                vocab_symbol_table.add_label(word)
                assert(vocab_symbol_table.get_label_id(word) == int(word_id))
        return vocab_symbol_table
