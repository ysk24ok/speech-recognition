class AcousticLabels(object):

    def get_label_id(self, label):
        """
        Args:
            label (str): A label
        Returns:
            (int): Label id
        """
        pass


class MonophoneLabels(AcousticLabels):

    def __init__(self, phonemes, kana2phonemes):
        self._basic_labels = ['<epsilon>', '<blank>']
        self._phonemes = phonemes
        self._auxiliary_labels = []
        self._kana2phonemes = kana2phonemes

    def get_epsilon_id(self):
        return self._basic_labels.index('<epsilon>')

    def get_blank_id(self):
        return self._basic_labels.index('<blank>')

    def get_label_id(self, label):
        offset = len(self._basic_labels)
        return self._phonemes.index(label) + offset

    def get_auxiliary_label_id(self, label):
        offset = len(self._basic_labels) + len(self._phonemes)
        return self._auxiliary_labels.index(label) + offset

    def set_auxiliary_label(self, label):
        if label not in self._auxiliary_labels:
            self._auxiliary_labels.append(label)

    def get_label(self, label_id):
        if label_id < len(self._basic_labels):
            return self._basic_labels[label_id]
        elif label_id < (len(self._basic_labels) + len(self._phonemes)):
            idx = label_id - len(self._basic_labels)
            return self._phonemes[idx]
        else:
            idx = label_id - len(self._basic_labels) - len(self._phonemes)
            return self._auxiliary_labels[idx]

    def get_label_ids(self, kana):
        label_ids = []
        for phoneme in self._kana2phonemes[kana].split():
            label_ids.append(self.get_label_id(phoneme))
        return label_ids

    def get_all_labels(self):
        labels = {idx: label for idx, label in enumerate(self._basic_labels)}
        offset = len(self._basic_labels)
        for idx, label in enumerate(self._phonemes):
            labels[idx + offset] = label
        return labels

    def get_all_auxiliary_labels(self):
        offset = len(self._basic_labels) + len(self._phonemes)
        return {idx + offset: label
                for idx, label in enumerate(self._auxiliary_labels)}

    def save(self, path):
        with open(path, 'w') as f:
            for idx, label in self.get_all_labels().items():
                entry = '{}\t{}\n'.format(label, idx)
                f.write(entry)
            for idx, label in self.get_all_auxiliary_labels().items():
                entry = '{}\t{}\n'.format(label, idx)
                f.write(entry)


class TriphoneLabels(AcousticLabels):

    pass


class MoraLabels(AcousticLabels):

    pass
