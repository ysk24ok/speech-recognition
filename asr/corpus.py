class Corpus(object):

    def __init__(self):
        self._corpus = []
        self._sentence = []

    def add(self, word, end_of_sentence=False):
        """Add a word to the corpus
        Args:
            word (str): a word
            end_of_sentence (bool): EOS flag
        """
        if end_of_sentence is True:
            self._sentence.append(word)
            self._corpus.append(self._sentence)
            self._sentence = []
        else:
            self._sentence.append(word)

    def save(self, path):
        """Save the corpus to disk
        Args:
            path (str): corpus filepath
        """
        with open(path, 'w') as f:
            sentences = [' '.join(sentence) for sentence in self._corpus]
            f.write('\n'.join(sentences))
