import math
import os
import shutil

import pytest
import pywrapfst

from asr import phonemes, kana2phonemes
from asr.decoder import (
    Corpus,
    _FstCompiler,
    Grammar,
    Lexicon,
    Token,
    WFSTDecoder
)
from asr.label_table import (
    PhonemeTable,
    VocabularyTable,
    VocabularySymbolTable
)


@pytest.fixture
def workdir():
    dirpath = '/tmp/work'
    os.mkdir(dirpath)
    yield dirpath
    shutil.rmtree(dirpath)


def is_expected_arc(arc, ilabel, olabel, nextstate, weight=None):
    assert(arc.ilabel == ilabel)
    assert(arc.olabel == olabel)
    assert(arc.nextstate == nextstate)
    if weight is not None:
        arc_weight = float(arc.weight.to_string().decode('utf-8'))
        assert(math.isclose(arc_weight, weight, rel_tol=1e-04))


def print_token_fst(fst, phoneme_table):
    for state_id in fst.states():
        is_final = fst.final(state_id).to_string().decode('utf-8') == '0'
        print('state: {}, is_final: {}'.format(state_id, is_final))
        for arc in fst.arcs(state_id):
            print(' ', state_id, arc.nextstate,
                  phoneme_table.get_label(arc.ilabel),
                  phoneme_table.get_label(arc.olabel),
                  arc.weight.to_string().decode('utf-8'))


def print_lexicon_fst(fst, phoneme_table, vocab):
    for state_id in fst.states():
        is_final = fst.final(state_id).to_string().decode('utf-8') == '0'
        print('state: {}, is_final: {}'.format(state_id, is_final))
        for arc in fst.arcs(state_id):
            print(' ', state_id, arc.nextstate,
                  phoneme_table.get_label(arc.ilabel),
                  vocab.get_word(arc.olabel),
                  arc.weight.to_string().decode('utf-8'))


def print_grammar_fst(fst, vocab):
    for state_id in fst.states():
        is_final = fst.final(state_id).to_string().decode('utf-8') == '0'
        print('state: {}, is_final: {}'.format(state_id, is_final))
        for arc in fst.arcs(state_id):
            print(' ', state_id, arc.nextstate,
                  vocab.get_word(arc.ilabel),
                  vocab.get_word(arc.olabel),
                  arc.weight.to_string().decode('utf-8'))


def test_token_create_fst():
    phoneme_table = PhonemeTable()
    phoneme_table.add_labels(['a', 'i'])
    epsilon_id = phoneme_table.get_epsilon_id()
    blank_id = phoneme_table.get_blank_id()
    a = phoneme_table.get_label_id('a')
    i = phoneme_table.get_label_id('i')

    fst = Token().create_fst(phoneme_table)
    assert(fst.num_states() == 5)
    # start state
    state = 0
    assert(fst.num_arcs(state) == 3)
    gen_arc = fst.arcs(state)
    is_expected_arc(next(gen_arc), blank_id, epsilon_id, state)
    is_expected_arc(next(gen_arc), a, a, 3)
    is_expected_arc(next(gen_arc), i, i, 4)
    # second state
    state = 1
    assert(fst.num_arcs(state) == 2)
    gen_arc = fst.arcs(state)
    is_expected_arc(next(gen_arc), blank_id, epsilon_id, state)
    is_expected_arc(next(gen_arc), epsilon_id, epsilon_id, 2)
    # final(auxiliary) state
    state = 2
    assert(fst.num_arcs(state) == 1)
    gen_arc = fst.arcs(state)
    is_expected_arc(next(gen_arc), epsilon_id, epsilon_id, 0)
    # a
    state = 3
    assert(fst.num_arcs(state) == 2)
    gen_arc = fst.arcs(state)
    is_expected_arc(next(gen_arc), a, epsilon_id, state)
    is_expected_arc(next(gen_arc), epsilon_id, epsilon_id, 1)
    # b
    state = 4
    assert(fst.num_arcs(state) == 2)
    gen_arc = fst.arcs(state)
    is_expected_arc(next(gen_arc), i, epsilon_id, state)
    is_expected_arc(next(gen_arc), epsilon_id, epsilon_id, 1)


def test_token_create_fst_with_auxiliary_labels():
    phoneme_table = PhonemeTable()
    phoneme_table.add_labels(['a', 'i'])
    epsilon_id = phoneme_table.get_epsilon_id()
    blank_id = phoneme_table.get_blank_id()
    a = phoneme_table.get_label_id('a')
    i = phoneme_table.get_label_id('i')
    phoneme_table.set_auxiliary_label('#0')
    phoneme_table.set_auxiliary_label('#1')
    aux0 = phoneme_table.get_auxiliary_label_id('#0')
    aux1 = phoneme_table.get_auxiliary_label_id('#1')

    fst = Token().create_fst(phoneme_table)
    assert(fst.num_states() == 5)
    # start state
    state = 0
    assert(fst.num_arcs(state) == 3)
    gen_arc = fst.arcs(state)
    is_expected_arc(next(gen_arc), blank_id, epsilon_id, state)
    is_expected_arc(next(gen_arc), a, a, 3)
    is_expected_arc(next(gen_arc), i, i, 4)
    # second state
    state = 1
    assert(fst.num_arcs(state) == 2)
    gen_arc = fst.arcs(state)
    is_expected_arc(next(gen_arc), blank_id, epsilon_id, state)
    is_expected_arc(next(gen_arc), epsilon_id, epsilon_id, 2)
    # final(auxiliary) state
    state = 2
    assert(fst.num_arcs(state) == 3)
    gen_arc = fst.arcs(state)
    is_expected_arc(next(gen_arc), epsilon_id, epsilon_id, 0)
    is_expected_arc(next(gen_arc), epsilon_id, aux0, state)
    is_expected_arc(next(gen_arc), epsilon_id, aux1, state)
    # a
    state = 3
    assert(fst.num_arcs(state) == 2)
    gen_arc = fst.arcs(state)
    is_expected_arc(next(gen_arc), a, epsilon_id, state)
    is_expected_arc(next(gen_arc), epsilon_id, epsilon_id, 1)
    # b
    state = 4
    assert(fst.num_arcs(state) == 2)
    gen_arc = fst.arcs(state)
    is_expected_arc(next(gen_arc), i, epsilon_id, state)
    is_expected_arc(next(gen_arc), epsilon_id, epsilon_id, 1)


@pytest.fixture
def words_without_homophones():
    return [
        {'word': '愛', 'kana': ['ア', 'イ']},
        {'word': '青', 'kana': ['ア', 'オ']}
    ]


@pytest.fixture
def words_with_homophones():
    return [
        {'word': '愛', 'kana': ['ア', 'イ']},
        {'word': '藍', 'kana': ['ア', 'イ']}
    ]


def get_vocabulary_table(workdir, words):
    vocab_table = VocabularyTable()
    for word in words:
        vocab_table.add_label(word['word'])
    return vocab_table


def get_lexicon(words):
    lexicon = Lexicon()
    for word in words:
        phonemes = []
        for k in word['kana']:
            for phoneme in kana2phonemes[k].split():
                phonemes.append(phoneme)
        lexicon.add(word['word'], phonemes)
    return lexicon


def test_lexicon_create_fst_without_homophones(workdir,
                                               words_without_homophones):
    vocab = get_vocabulary_table(workdir, words_without_homophones)
    lexicon = get_lexicon(words_without_homophones)

    phoneme_table = PhonemeTable()
    phoneme_table.add_labels(phonemes)
    epsilon_id = phoneme_table.get_epsilon_id()
    a = phoneme_table.get_label_id('a')
    i = phoneme_table.get_label_id('i')
    o = phoneme_table.get_label_id('o')

    fst = lexicon.create_fst(phoneme_table, vocab, min_freq=0)
    assert(fst.num_states() == 7)
    aux0 = phoneme_table.get_auxiliary_label_id('#0')

    state = 0
    assert(fst.num_arcs(0) == 2)
    gen = fst.arcs(state)
    arc = next(gen)
    is_expected_arc(arc, a, vocab.get_label_id('愛'), 1)
    arc = gen.__next__()
    is_expected_arc(arc, a, vocab.get_label_id('青'), 4)

    state = 1
    assert(fst.num_arcs(state) == 1)
    arc = next(fst.arcs(state))
    is_expected_arc(arc, i, epsilon_id, 2)

    state = 2
    assert(fst.num_arcs(state) == 1)
    arc = next(fst.arcs(state))
    is_expected_arc(arc, aux0, epsilon_id, 3)

    state = 3
    assert(fst.num_arcs(state) == 1)
    arc = next(fst.arcs(state))
    is_expected_arc(arc, epsilon_id, epsilon_id, 0)

    state = 4
    assert(fst.num_arcs(state) == 1)
    arc = next(fst.arcs(state))
    is_expected_arc(arc, o, epsilon_id, 5)

    state = 5
    assert(fst.num_arcs(state) == 1)
    arc = next(fst.arcs(state))
    is_expected_arc(arc, aux0, epsilon_id, 6)

    state = 6
    assert(fst.num_arcs(state) == 1)
    arc = next(fst.arcs(state))
    is_expected_arc(arc, epsilon_id, epsilon_id, 0)


def test_lexicon_create_fst_with_homophones(workdir, words_with_homophones):
    vocab = get_vocabulary_table(workdir, words_with_homophones)
    lexicon = get_lexicon(words_with_homophones)

    phoneme_table = PhonemeTable()
    phoneme_table.add_labels(phonemes)
    epsilon_id = phoneme_table.get_epsilon_id()
    a = phoneme_table.get_label_id('a')
    i = phoneme_table.get_label_id('i')

    fst = lexicon.create_fst(phoneme_table, vocab, min_freq=0)
    assert(fst.num_states() == 7)
    aux0 = phoneme_table.get_auxiliary_label_id('#0')
    aux1 = phoneme_table.get_auxiliary_label_id('#1')

    state = 0
    assert(fst.num_arcs(state) == 2)
    gen_arc = fst.arcs(state)
    arc = next(gen_arc)
    is_expected_arc(arc, a, vocab.get_label_id('愛'), 1, weight=0.6931)
    arc = next(gen_arc)
    is_expected_arc(arc, a, vocab.get_label_id('藍'), 4, weight=0.6931)

    state = 1
    assert(fst.num_arcs(state) == 1)
    arc = next(fst.arcs(state))
    is_expected_arc(arc, i, epsilon_id, 2)

    state = 2
    assert(fst.num_arcs(state) == 1)
    arc = next(fst.arcs(state))
    is_expected_arc(arc, aux0, epsilon_id, 3)

    state = 3
    assert(fst.num_arcs(state) == 1)
    arc = next(fst.arcs(state))
    is_expected_arc(arc, epsilon_id, epsilon_id, 0)

    state = 4
    assert(fst.num_arcs(state) == 1)
    arc = next(fst.arcs(state))
    is_expected_arc(arc, i, epsilon_id, 5)

    state = 5
    assert(fst.num_arcs(state) == 1)
    arc = next(fst.arcs(state))
    is_expected_arc(arc, aux1, epsilon_id, 6)

    state = 6
    assert(fst.num_arcs(state) == 1)
    arc = next(fst.arcs(state))
    is_expected_arc(arc, epsilon_id, epsilon_id, 0)


def test_compose_token_and_lexicon_fst(workdir, words_without_homophones):
    vocab = get_vocabulary_table(workdir, words_without_homophones)
    lexicon = get_lexicon(words_without_homophones)

    phoneme_table = PhonemeTable()
    phoneme_table.add_labels(phonemes)

    lexicon_fst = lexicon.create_fst(phoneme_table, vocab)

    token = Token()
    token_fst = token.create_fst(phoneme_table)

    fst = pywrapfst.compose(token_fst.arcsort('olabel'), lexicon_fst)
    fst = pywrapfst.determinize(fst)


def test_compose_token_and_lexicon_fst_with_homophones(
        workdir, words_with_homophones):
    vocab = get_vocabulary_table(workdir, words_with_homophones)
    lexicon = get_lexicon(words_with_homophones)

    phoneme_table = PhonemeTable()
    phoneme_table.add_labels(phonemes)

    lexicon_fst = lexicon.create_fst(phoneme_table, vocab, min_freq=0)

    token = Token()
    token_fst = token.create_fst(phoneme_table)

    fst = pywrapfst.compose(token_fst.arcsort('olabel'), lexicon_fst)
    with pytest.raises(pywrapfst.FstOpError):
        pywrapfst.determinize(fst)


@pytest.fixture
def words_for_corpus_without_homophones():
    return [
        {'word': 'あなた', 'kana': ['ア', 'ナ', 'タ'], 'eos': False},
        {'word': 'を', 'kana': ['ヲ'], 'eos': False},
        {'word': '愛し', 'kana': ['ア', 'イ', 'シ'], 'eos': False},
        {'word': 'て', 'kana': ['テ'], 'eos': False},
        {'word': 'いる', 'kana': ['イ', 'ル'], 'eos': True},
        {'word': '藍', 'kana': ['ア', 'イ'], 'eos': False},
        {'word': 'で', 'kana': ['デ'], 'eos': False},
        {'word': '服', 'kana': ['フ', 'ク'], 'eos': False},
        {'word': 'を', 'kana': ['ヲ'], 'eos': False},
        {'word': '染める', 'kana': ['ソ', 'メ', 'ル'], 'eos': True},
    ]


@pytest.fixture
def words_for_corpus_with_homophones():
    return [
        {'word': '愛', 'kana': ['ア', 'イ'], 'eos': False},
        {'word': 'は', 'kana': ['ハ'], 'eos': False},
        {'word': '世界', 'kana': ['セ', 'カ', 'イ'], 'eos': False},
        {'word': 'を', 'kana': ['ヲ'], 'eos': False},
        {'word': '救う', 'kana': ['ス', 'ク', 'ウ'], 'eos': True},
        {'word': '藍', 'kana': ['ア', 'イ'], 'eos': False},
        {'word': 'で', 'kana': ['デ'], 'eos': False},
        {'word': '服', 'kana': ['フ', 'ク'], 'eos': False},
        {'word': 'を', 'kana': ['ヲ'], 'eos': False},
        {'word': '染める', 'kana': ['ソ', 'メ', 'ル'], 'eos': True},
    ]


def create_corpus(path, words):
    corpus = Corpus()
    for word in words:
        corpus.add(word['word'], end_of_sentence=word['eos'])
    corpus.save(path)


def create_vocabulary_symbol_table(path, corpus_path):
    VocabularySymbolTable.create_symbol(path, corpus_path)
    vocab = VocabularySymbolTable.load_symbol(path)
    return vocab


def test_grammar_create_fst(workdir, words_for_corpus_without_homophones):
    corpus_path = os.path.join(workdir, 'corpus.txt')
    create_corpus(corpus_path, words_for_corpus_without_homophones)

    vocab_path = os.path.join(workdir, 'vocab.syms')
    create_vocabulary_symbol_table(vocab_path, corpus_path)

    grammar_path = os.path.join(workdir, 'grammar.fst')
    grammar = Grammar()
    grammar.create_fst(grammar_path, vocab_path, corpus_path)


def test_wfst_decoder_create_fst(workdir, words_for_corpus_without_homophones):
    corpus_path = os.path.join(workdir, 'corpus.txt')
    create_corpus(corpus_path, words_for_corpus_without_homophones)

    vocab_path = os.path.join(workdir, 'vocab.syms')
    vocab = create_vocabulary_symbol_table(vocab_path, corpus_path)

    phoneme_table = PhonemeTable()
    phoneme_table.add_labels(phonemes)

    lexicon = get_lexicon(words_for_corpus_without_homophones)
    lexicon_fst = lexicon.create_fst(phoneme_table, vocab, min_freq=0)

    token = Token()
    token_fst = token.create_fst(phoneme_table)

    grammar_path = os.path.join(workdir, 'grammar.fst')
    grammar = Grammar()
    grammar_fst = grammar.create_fst(grammar_path, vocab_path, corpus_path)

    wfst_decoder = WFSTDecoder()
    wfst_decoder.create_fst(token_fst, lexicon_fst, grammar_fst)


def test_wfst_decoder_create_fst_with_homophones(
        workdir, words_for_corpus_with_homophones):
    corpus_path = os.path.join(workdir, 'corpus.txt')
    create_corpus(corpus_path, words_for_corpus_with_homophones)

    vocab_path = os.path.join(workdir, 'vocab.syms')
    vocab = create_vocabulary_symbol_table(vocab_path, corpus_path)

    phoneme_table = PhonemeTable()
    phoneme_table.add_labels(phonemes)

    lexicon = get_lexicon(words_for_corpus_with_homophones)
    lexicon_fst = lexicon.create_fst(phoneme_table, vocab, min_freq=0)

    token = Token()
    token_fst = token.create_fst(phoneme_table)

    grammar_path = os.path.join(workdir, 'grammar.fst')
    grammar = Grammar()
    grammar_fst = grammar.create_fst(grammar_path, vocab_path, corpus_path)

    wfst_decoder = WFSTDecoder()
    wfst_decoder.create_fst(token_fst, lexicon_fst, grammar_fst)


def test_wfst_decoder_decode(workdir, words_for_corpus_with_homophones):
    corpus_path = os.path.join(workdir, 'corpus.txt')
    create_corpus(corpus_path, words_for_corpus_with_homophones)

    vocab_path = os.path.join(workdir, 'vocab.syms')
    vocab = create_vocabulary_symbol_table(vocab_path, corpus_path)

    phoneme_table = PhonemeTable()
    phoneme_table.add_labels(phonemes)

    lexicon = get_lexicon(words_for_corpus_with_homophones)
    lexicon_fst = lexicon.create_fst(phoneme_table, vocab, min_freq=0)

    token = Token()
    token_fst = token.create_fst(phoneme_table)

    grammar_path = os.path.join(workdir, 'grammar.fst')
    grammar = Grammar()
    grammar_fst = grammar.create_fst(grammar_path, vocab_path, corpus_path)

    wfst_decoder = WFSTDecoder()
    wfst_decoder.create_fst(token_fst, lexicon_fst, grammar_fst)

    blank_id = phoneme_table.get_blank_id()
    a = phoneme_table.get_label_id('a')
    i = phoneme_table.get_label_id('i')
    d = phoneme_table.get_label_id('d')
    e = phoneme_table.get_label_id('e')
    s = phoneme_table.get_label_id('s')
    o = phoneme_table.get_label_id('o')
    m = phoneme_table.get_label_id('m')
    r = phoneme_table.get_label_id('r')
    u = phoneme_table.get_label_id('u')
    frame_labels = [blank_id, blank_id, a, a, i, i, i, d, e, blank_id,
                    s, s, o, o, o, m, e, r, r, u]
    got = wfst_decoder.decode(frame_labels, vocab)
    assert got == '藍で染める'


def test_wfst_decoder_epsilon_transition():
    phoneme_table = PhonemeTable()
    phoneme_table.add_labels(phonemes)

    fst_compiler = _FstCompiler()
    eps = phoneme_table.get_epsilon_id()
    a = phoneme_table.get_label_id('a')
    fst_compiler.add_arc(0, 1, eps, eps, 0.1)
    fst_compiler.add_arc(1, 2, eps, eps, 0.2)
    fst_compiler.add_arc(1, 3, eps, eps, 0.3)
    fst_compiler.add_arc(0, 2, eps, eps, 0.15)
    fst_compiler.add_arc(0, 3, eps, eps, 0.5)
    fst_compiler.add_arc(0, 4, a, eps, 0.15)
    fst_compiler.set_final(4)
    fst = fst_compiler.compile()

    wfst_decoder = WFSTDecoder(fst)
    paths = {
        0: wfst_decoder.Path(
            score=0, prev_path=None, frame_index=0, olabel=None)
    }
    frame_index = 0
    wfst_decoder.epsilon_transition(paths, phoneme_table, frame_index)
    # check new state 1 is added to paths
    # TODO: state:1が残るのは果たしてよいのか？無限ループしそう
    assert 1 in paths
    assert round(paths[1].score, 6) == 0.1
    assert paths[1].prev_path.score == 0
    # check existing state 2 is updated via better path
    assert 2 in paths
    assert round(paths[2].score, 6) == 0.15
    assert paths[2].prev_path.score == 0
    # check existing state 3 is not updated
    assert round(paths[3].score, 6) == 0.4
    assert round(paths[3].prev_path.score, 6) == 0.1
    assert paths[3].prev_path.prev_path.score == 0
    # check new state 4 is not added to paths
    # because it's not epsilon transition
    assert 4 not in paths


def test_wfst_decoder_normal_transition():
    phoneme_table = PhonemeTable()
    phoneme_table.add_labels(phonemes)

    fst_compiler = _FstCompiler()
    eps = phoneme_table.get_epsilon_id()
    blank = phoneme_table.get_blank_id()
    a = phoneme_table.get_label_id('a')
    i = phoneme_table.get_label_id('i')
    fst_compiler.add_arc(0, 1, blank, eps, 0.2)
    fst_compiler.add_arc(1, 2, a, eps, 0.1)
    fst_compiler.add_arc(1, 3, i, eps, 0.2)
    fst = fst_compiler.compile()

    wfst_decoder = WFSTDecoder(fst)
    prev_paths = {
        0: wfst_decoder.Path(
            score=0, prev_path=None, frame_index=0, olabel=None)
    }
    curr_paths = {}
    wfst_decoder.normal_transition(prev_paths, curr_paths, 0, blank)
    assert 1 in curr_paths
    assert round(curr_paths[1].score, 6) == 0.2
    assert round(curr_paths[1].prev_path.score, 6) == 0
    prev_paths = curr_paths
    curr_paths = {}
    wfst_decoder.normal_transition(prev_paths, curr_paths, 1, a)
    assert 2 in curr_paths
    assert round(curr_paths[2].score, 6) == 0.3
    assert curr_paths[2].frame_index == 1
    assert round(curr_paths[2].prev_path.score, 6) == 0.2
    # TODO: no tests for better path


def test_wfst_decoder_prune_paths():
    wfst_decoder = WFSTDecoder()
    paths = {
        1: WFSTDecoder.Path(
            score=2.0, prev_path=None, frame_index=0, olabel=None),
        2: WFSTDecoder.Path(
            score=5.0, prev_path=None, frame_index=0, olabel=None),
        3: WFSTDecoder.Path(
            score=1.0, prev_path=None, frame_index=0, olabel=None),
        4: WFSTDecoder.Path(
            score=4.0, prev_path=None, frame_index=0, olabel=None),
        5: WFSTDecoder.Path(
            score=3.0, prev_path=None, frame_index=0, olabel=None),
    }
    beamwidth = 3
    got = wfst_decoder.prune_paths(paths, beamwidth=beamwidth)
    assert(len(got) == beamwidth)
    assert 1 in got
    assert 3 in got
    assert 5 in got
