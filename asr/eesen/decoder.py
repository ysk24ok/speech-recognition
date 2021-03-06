import os
import subprocess
from collections import defaultdict, namedtuple
from queue import Queue

import numpy
import pywrapfst


class _FstCompiler(object):

    def __init__(self):
        self._compiler = pywrapfst.Compiler()

    def add_arc(self, start_state, dest_state, ilabel, olabel, weight=None):
        if weight is not None:
            entry = '{} {} {} {} {}'.format(
                start_state, dest_state, ilabel, olabel, weight)
        else:
            entry = '{} {} {} {}'.format(
                start_state, dest_state, ilabel, olabel)
        print(entry, file=self._compiler)

    def set_final(self, state_id):
        print(state_id, file=self._compiler)

    def compile(self):
        return self._compiler.compile()


class Token(object):

    """Maintain a mappping from acoustic labels to a single lexicon unit"""

    def create_fst(self, phoneme_table):
        """Create a token FST
        Args:
            phoneme_table (asr.label_table.PhonemeTable):
                PhonemeTable object
        Returns:
            pywrapfst._MutableFst: token FST
        """
        fst_compiler = _FstCompiler()
        epsilon_id = phoneme_table.get_epsilon_id()
        blank_id = phoneme_table.get_blank_id()
        # start state
        start_state_id = 0
        fst_compiler.add_arc(
            start_state_id, start_state_id, blank_id, epsilon_id)
        # second state
        second_state_id = start_state_id + 1
        fst_compiler.add_arc(
            second_state_id, second_state_id, blank_id, epsilon_id)
        # final state
        final_state_id = second_state_id + 1
        fst_compiler.add_arc(
            second_state_id, final_state_id, epsilon_id, epsilon_id)
        fst_compiler.add_arc(
            final_state_id, start_state_id, epsilon_id, epsilon_id)
        fst_compiler.set_final(final_state_id)
        state_id = final_state_id + 1
        for label_id, label in phoneme_table.get_all_labels().items():
            if label in ('<epsilon>', '<blank>'):
                continue
            fst_compiler.add_arc(
                start_state_id, state_id, label_id, label_id)
            fst_compiler.add_arc(
                state_id, state_id, label_id, epsilon_id)
            fst_compiler.add_arc(
                state_id, second_state_id, epsilon_id, epsilon_id)
            state_id += 1
        for label_id, _ in phoneme_table.get_all_auxiliary_labels().items():
            fst_compiler.add_arc(
                final_state_id, final_state_id, epsilon_id, label_id)
        return fst_compiler.compile()


class Lexicon(object):

    """Maintain a mapping from sequences of lexicon units to words
    Parameters:
        _lexicon (dict[str, dict[int]]):
            mapping whose key is a word and value is list of phoneme id
        _homophone (dict[str, list[str]]):
            mapping whose key is pronunciation and value is a word list
        _total_word_count (int): total word count
    """

    def __init__(self):
        self._lexicon = defaultdict(lambda: defaultdict(int))
        self._total_word_count = 0
        self._homophone = defaultdict(lambda: [])

    def add(self, word, lexicon_units):
        """Add a word and its pronunciation to the lexicon
        Args:
            word (str): A word
            lexicon_units (list[str]): Lexicon units (phonemes or moras)
        """
        lexicon_units_concat = ' '.join(lexicon_units)
        if lexicon_units_concat == '':
            return
        self._lexicon[word][lexicon_units_concat] += 1
        if word not in self._homophone[lexicon_units_concat]:
            self._homophone[lexicon_units_concat].append(word)
        self._total_word_count += 1

    def save(self, path):
        """Save a lexicon to disk
        Args:
            path (str): Lexicon filepath
        """
        lines = []
        for word, freq_per_lexicon_units in self._lexicon.items():
            for lexicon_units_concat, freq in freq_per_lexicon_units.items():
                line = '\t'.join([word, lexicon_units_concat, str(freq)])
                lines.append('{}\n'.format(line))
        with open(path, 'w') as f:
            f.writelines(lines)

    def load(self, path):
        """Load a lexcion file from disk
        Args:
            path (str): Lexicon filepath
        """
        with open(path) as f:
            for line in f:
                word, lexicon_units_concat, freq = line.split('\t')
                if lexicon_units_concat == '':
                    continue
                freq = int(freq)
                self._lexicon[word][lexicon_units_concat] += freq
                self._homophone[lexicon_units_concat].append(word)
                self._total_word_count += freq

    def remove_less_frequent_word(self, min_freq):
        pairs = []
        for word, freq_per_lexicon_units in self._lexicon.items():
            for lexicon_units_concat, freq in freq_per_lexicon_units.items():
                if freq < min_freq:
                    pairs.append((word, lexicon_units_concat))
        for word, lexicon_units_concat in pairs:
            freq = self._lexicon[word][lexicon_units_concat]
            del self._lexicon[word][lexicon_units_concat]
            self._total_word_count -= freq

    def create_fst(self, label_table, vocabulary_table, min_freq=10):
        """Create a lexicon FST
        Args:
            label_table (asr.label_table.LabelTable):
                LabelTable object
            vocabulary_table (asr.label_table.VocabularyTable):
                VocabularyTable object
            min_freq (int): Minimum frequency for a word
                            to be included in lexicon
        Returns:
            pywrapfst._MutableFst: A lexicon FST
        """
        self.remove_less_frequent_word(min_freq)
        fst_compiler = _FstCompiler()
        epsilon_id = label_table.get_epsilon_id()
        start_state_id = 0
        fst_compiler.set_final(start_state_id)
        state_id = start_state_id + 1
        for word, freq_per_lexicon_units in self._lexicon.items():
            word_id = vocabulary_table.get_label_id(word)
            for lexicon_units_concat, freq in freq_per_lexicon_units.items():
                prob = -numpy.log(freq / self._total_word_count)
                idx = 0
                for lexicon_unit in lexicon_units_concat.split():
                    label_id = label_table.get_label_id(lexicon_unit)
                    if idx == 0:
                        fst_compiler.add_arc(
                            start_state_id, state_id, label_id, word_id,
                            prob)
                    else:
                        fst_compiler.add_arc(
                            state_id - 1, state_id, label_id, epsilon_id)
                    state_id += 1
                    idx += 1
                # When there is no homophone, homophone_word_id will be 0
                homophone_word_id = \
                    self._homophone[lexicon_units_concat].index(word)
                aux_label = '#{}'.format(homophone_word_id)
                label_table.set_auxiliary_label(aux_label)
                label_id = label_table.get_auxiliary_label_id(
                    aux_label)
                fst_compiler.add_arc(state_id - 1, state_id,
                                     label_id, epsilon_id)
                fst_compiler.add_arc(state_id, start_state_id,
                                     epsilon_id, epsilon_id)
                state_id += 1
        return fst_compiler.compile()


class Grammar(object):

    def create_fst(self, path, vocab_path, corpus_path, order=2):
        """Create a grammar FST
        Args:
            path (str): grammar FST filepath
            vocab_path (str): vocabulary symbol filepath
            corpus_path (str): corpus filepath
            order (int): maximal order of ngrams to be counted
        """
        vocab_dirpath, vocab_filename = os.path.split(vocab_path)
        # NOTE: Exec chdir to pass vocabulary symbol file name (not file path)
        #       to --symbols option, otherwise created .far won't be identical
        cwd = os.getcwd()
        os.chdir(vocab_dirpath)
        p1 = subprocess.Popen([
                'farcompilestrings',
                '--symbols={}'.format(vocab_filename),
                '--keep_symbols=1',
                corpus_path
            ],
            stdout=subprocess.PIPE
        )
        p2 = subprocess.Popen([
                'ngramcount',
                '--order={}'.format(order)
            ],
            stdin=p1.stdout,
            stdout=subprocess.PIPE
        )
        p3 = subprocess.Popen([
                'ngrammake',
            ],
            stdin=p2.stdout,
            stdout=subprocess.PIPE
        )
        p4 = subprocess.Popen([
                'ngramshrink',
                '-',
                path
            ],
            stdin=p3.stdout,
        )
        p4.wait()
        os.chdir(cwd)
        return pywrapfst.Fst.read(path)


class WFSTDecoder(object):

    """
    Parameters:
        _fst (pywrapfst._MutableFst): A FST
    """

    Path = namedtuple('Path', ['score', 'prev_path', 'frame_index', 'olabel'])

    def __init__(self, fst=None):
        self._fst = fst

    def create_fst(self, token_fst, lexicon_fst, grammar_fst):
        """Create FST by composing token, lexicon and grammar FST
        Args:
            token_fst (pywrapfst._MutableFst): A token FST
            lexicon_fst (pywrapfst._MutableFst): A lexicon FST
            grammar_fst (pywrapfst._MutableFst): A grammar FST
        """
        LG = pywrapfst.determinize(
            pywrapfst.compose(lexicon_fst, grammar_fst.arcsort()).rmepsilon()
        ).minimize()
        self._fst = pywrapfst.compose(token_fst, LG.arcsort())

    def read_fst(self, path):
        """Read a FST from disk
        Args:
            path (str): FST path
        """
        self._fst = pywrapfst.Fst.read(path)

    def write_fst(self, path):
        """Write a FST to disk
        Args:
            path (str): FST path
        """
        self._fst.write(path)

    def decode(self, frame_labels, vocabulary_symbol, beamwidth=5,
               epsilon_id=0):
        """Decode acoustic fram labels into string
        Args:
            frame_labels (list[int]): labels per frames
            vocabulary_symbol (asr.decoder.VocabularySymbol):
                vocabulary symbol table
            beamwidth (int): maximum number of states to maintain
                             in beam search
            epsilon_id (int): epsilon id of acoustic labels
        Returns:
            (str): decoded string
        """
        prev_paths = {}
        prev_paths[self._fst.start()] = self.Path(
            score=0, prev_path=None, frame_index=0, olabel=None)
        for frame_index, frame_label in enumerate(frame_labels):
            curr_paths = {}
            self.epsilon_transition(prev_paths, frame_index, ieps=epsilon_id)
            self.normal_transition(prev_paths, curr_paths,
                                   frame_index, frame_label)
            prev_paths = self.prune_paths(curr_paths, beamwidth=beamwidth)
        best_path = self.get_best_path(prev_paths)
        return ''.join(self.get_words(best_path, vocabulary_symbol))

    def epsilon_transition(self, prev_paths, frame_index, ieps=0):
        q = Queue()
        for state in prev_paths.keys():
            q.put(state)
        while q.empty() is False:
            state = q.get()
            path = prev_paths[state]
            for arc in self._fst.arcs(state):
                if arc.ilabel != ieps:
                    continue
                weight = float(arc.weight.to_string().decode('utf-8'))
                new_score = weight + path.score
                new_path = self.Path(
                    score=new_score,
                    prev_path=path,
                    frame_index=frame_index,
                    olabel=arc.olabel)
                if arc.nextstate not in prev_paths:   # transit to new state
                    prev_paths[arc.nextstate] = new_path
                    q.put(arc.nextstate)
                else:
                    # transit to the existing state via better path
                    if new_score < prev_paths[arc.nextstate].score:
                        prev_paths[arc.nextstate] = new_path
                        q.put(arc.nextstate)

    def normal_transition(self, prev_paths, curr_paths,
                          frame_index, frame_label):
        for state, path in prev_paths.items():
            for arc in self._fst.arcs(state):
                if arc.ilabel != frame_label:
                    continue
                weight = float(arc.weight.to_string().decode('utf-8'))
                new_score = weight + path.score
                new_path = self.Path(
                    score=new_score, prev_path=path,
                    frame_index=frame_index, olabel=arc.olabel)
                if arc.nextstate not in curr_paths:     # transit to new state
                    curr_paths[arc.nextstate] = new_path
                else:
                    # transit to the existing state via better path
                    if new_score < curr_paths[arc.nextstate].score:
                        curr_paths[arc.nextstate] = new_path

    def prune_paths(self, paths, beamwidth=5):
        sorted_paths = sorted(
            list(paths.items()), key=lambda x: x[1].score)
        return dict(sorted_paths[0:beamwidth])

    def backtrack(self, path):
        olabels = []
        while path is not None:
            if path.olabel is not None:
                olabels.insert(0, path.olabel)
            path = path.prev_path
        return olabels

    def get_words(self, path, vocabulary_table):
        epsilon_id = vocabulary_table.get_epsilon_id()
        words = []
        for olabel in self.backtrack(path):
            if olabel == epsilon_id:
                continue
            words.append(vocabulary_table.get_label(olabel))
        return words

    def get_best_path(self, paths):
        best_path = None
        best_score = -numpy.log(numpy.finfo(float).eps)
        for state, path in paths.items():
            if best_path is None or path.score < best_score:
                best_path = path
                best_score = path.score
        return best_path
