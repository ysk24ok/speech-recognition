class Token(object):

    """Maintain a mappping from CTC labels to a single lexicon unit"""
    pass


class Lexicon(object):

    """Maintain a mapping from sequences of lexicon units to words
    Parameters:
        kana_phoneme_mapping (dict[str, str]):
            mapping whose key is kana and value is phoneme
        phonemes (list[str]): phoneme list
        _lexicon (dict[str, list[int]]):
            mapping whose key is a word and value is list of phoneme id
    """

    def __init__(self, phonemes, kana_phoneme_mapping):
        self.phonemes = phonemes
        self.kana_phoneme_mapping = kana_phoneme_mapping
        self._lexicon = {}

    def add(self, word, reading):
        """Add a word to the lexicon entry
        Arguments:
            word (str): a word
            reading (list[str]): mora list of the word
        """
        if word in self._lexicon:
            return
        self._lexicon[word] = []
        for mora in reading:
            for phoneme in self.kana_phoneme_mapping[mora].split():
                phoneme_id = self.phonemes.index(phoneme)
                self._lexicon[word].append(phoneme_id)

    def get(self, word):
        return self._lexicon[word]

    def save(self, path):
        pass

    def load(self, path):
        pass
