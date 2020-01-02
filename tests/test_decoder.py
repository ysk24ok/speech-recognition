import math
import os
import shutil

import pytest
import pywrapfst

from asr import phonemes, kana2phonemes
from asr.acoustic_labels import MonophoneLabels
from asr.decoder import (
    Corpus,
    _FstCompiler,
    Grammar,
    Lexicon,
    Token,
    VocabularySymbol,
    WFSTDecoder
)


@pytest.fixture
def workdir():
    dirpath = '/tmp/work'
    os.mkdir(dirpath)
    yield dirpath
    shutil.rmtree(dirpath)


def test_vocabulary_symbol_create(workdir):
    corpus_path = os.path.join(workdir, 'corpus.txt')
    corpus = Corpus()
    corpus.add('ボール', end_of_sentence=False)
    corpus.add('を', end_of_sentence=False)
    corpus.add('投げる', end_of_sentence=True)
    corpus.save(corpus_path)

    vocab_path = os.path.join(workdir, 'vocab.syms')
    vocab = VocabularySymbol()
    vocab.create(vocab_path, corpus_path)
    vocab.read(vocab_path)
    assert(vocab.get_id('<epsilon>') == 0)
    assert(vocab.get_word(0) == '<epsilon>')
    assert(vocab.get_id('ボール') == 1)
    assert(vocab.get_word(1) == 'ボール')
    assert(vocab.get_id('を') == 2)
    assert(vocab.get_word(2) == 'を')
    assert(vocab.get_id('投げる') == 3)
    assert(vocab.get_word(3) == '投げる')
    assert(vocab.get_id('<unk>') == 4)
    assert(vocab.get_word(4) == '<unk>')


def is_expected_arc(arc, ilabel, olabel, nextstate, weight=None):
    assert(arc.ilabel == ilabel)
    assert(arc.olabel == olabel)
    assert(arc.nextstate == nextstate)
    if weight is not None:
        arc_weight = float(arc.weight.to_string().decode('utf-8'))
        assert(math.isclose(arc_weight, weight, rel_tol=1e-04))


def print_token_fst(fst, acoustic_labels):
    for state_id in fst.states():
        is_final = fst.final(state_id).to_string().decode('utf-8') == '0'
        print('state: {}, is_final: {}'.format(state_id, is_final))
        for arc in fst.arcs(state_id):
            print(' ', state_id, arc.nextstate,
                  acoustic_labels.get_label(arc.ilabel),
                  acoustic_labels.get_label(arc.olabel),
                  arc.weight.to_string().decode('utf-8'))


def print_lexicon_fst(fst, acoustic_labels, vocab):
    for state_id in fst.states():
        is_final = fst.final(state_id).to_string().decode('utf-8') == '0'
        print('state: {}, is_final: {}'.format(state_id, is_final))
        for arc in fst.arcs(state_id):
            print(' ', state_id, arc.nextstate,
                  acoustic_labels.get_label(arc.ilabel),
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


@pytest.fixture
def acoustic_labels():
    phonemes = ['a', 'i']
    kana2phonemes = {'ア': 'a', 'イ': 'i'}
    return MonophoneLabels(phonemes, kana2phonemes)


def test_token_create_fst(acoustic_labels):
    epsilon_id = acoustic_labels.get_epsilon_id()
    blank_id = acoustic_labels.get_blank_id()
    a = acoustic_labels.get_label_id('a')
    i = acoustic_labels.get_label_id('i')

    token = Token()
    fst = token.create_fst(acoustic_labels)
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


def test_token_create_fst_with_auxiliary_labels(acoustic_labels):
    epsilon_id = acoustic_labels.get_epsilon_id()
    blank_id = acoustic_labels.get_blank_id()
    a = acoustic_labels.get_label_id('a')
    i = acoustic_labels.get_label_id('i')
    acoustic_labels.set_auxiliary_label('#0')
    acoustic_labels.set_auxiliary_label('#1')
    aux0 = acoustic_labels.get_auxiliary_label_id('#0')
    aux1 = acoustic_labels.get_auxiliary_label_id('#1')

    token = Token()
    fst = token.create_fst(acoustic_labels)
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
def lexicon():
    phonemes = ['u', 'ky', 'sh', ':']
    kana2phonemes = {
        'ウ': 'u',
        'キュ': 'ky u',
        'シュ': 'sh u',
        'ー': ':'
    }
    acoustic_labels = MonophoneLabels(phonemes, kana2phonemes)
    return Lexicon(acoustic_labels)


def test_lexicon_get_label_ids(lexicon):
    reading = ['キュ', 'ー', 'シュ', 'ー']
    got = lexicon.get_label_ids(reading)
    expected = [3, 2, 5, 4, 2, 5]
    assert(got == expected)


def get_vocabulary_symbol(workdir, words):
    corpus_path = os.path.join(workdir, 'corpus.txt')
    corpus = Corpus()
    for word in words:
        corpus.add(word['word'], end_of_sentence=word['eos'])
    corpus.save(corpus_path)
    vocab_path = os.path.join(workdir, 'vocab.syms')
    vocab = VocabularySymbol()
    vocab.create(vocab_path, corpus_path)
    vocab.read(vocab_path)
    return vocab


def test_lexicon_create_fst_without_homophones(workdir):
    words = [
        {'word': '愛', 'kana': ['ア', 'イ'], 'eos': False},
        {'word': '青', 'kana': ['ア', 'オ'], 'eos': True}
    ]
    vocab = get_vocabulary_symbol(workdir, words)

    acoustic_labels = MonophoneLabels(phonemes, kana2phonemes)
    lexicon = Lexicon(acoustic_labels)
    for word in words:
        lexicon.add(word['word'], word['kana'])

    epsilon_id = acoustic_labels.get_epsilon_id()
    a = acoustic_labels.get_label_id('a')
    i = acoustic_labels.get_label_id('i')
    o = acoustic_labels.get_label_id('o')

    fst = lexicon.create_fst(vocab, min_freq=0)
    assert(fst.num_states() == 7)
    aux0 = acoustic_labels.get_auxiliary_label_id('#0')

    state = 0
    assert(fst.num_arcs(0) == 2)
    gen = fst.arcs(state)
    arc = next(gen)
    is_expected_arc(arc, a, vocab.get_id('愛'), 1)
    arc = gen.__next__()
    is_expected_arc(arc, a, vocab.get_id('青'), 4)

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


def test_lexicon_create_fst_with_homophones(workdir):
    words = [
        {'word': '愛', 'kana': ['ア', 'イ'], 'eos': False},
        {'word': '藍', 'kana': ['ア', 'イ'], 'eos': True}
    ]
    vocab = get_vocabulary_symbol(workdir, words)

    acoustic_labels = MonophoneLabels(phonemes, kana2phonemes)
    lexicon = Lexicon(acoustic_labels)
    for word in words:
        lexicon.add(word['word'], word['kana'])

    epsilon_id = acoustic_labels.get_epsilon_id()
    a = acoustic_labels.get_label_id('a')
    i = acoustic_labels.get_label_id('i')

    fst = lexicon.create_fst(vocab, min_freq=0)
    assert(fst.num_states() == 7)
    aux0 = acoustic_labels.get_auxiliary_label_id('#0')
    aux1 = acoustic_labels.get_auxiliary_label_id('#1')

    state = 0
    assert(fst.num_arcs(state) == 2)
    gen_arc = fst.arcs(state)
    arc = next(gen_arc)
    is_expected_arc(arc, a, vocab.get_id('愛'), 1, weight=0.6931)
    arc = next(gen_arc)
    is_expected_arc(arc, a, vocab.get_id('藍'), 4, weight=0.6931)

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


def test_compose_token_and_lexicon_fst(workdir):
    words = [
        {'word': '愛', 'kana': ['ア', 'イ'], 'eos': False},
        {'word': '青', 'kana': ['ア', 'オ'], 'eos': True}
    ]
    vocab = get_vocabulary_symbol(workdir, words)

    acoustic_labels = MonophoneLabels(phonemes, kana2phonemes)
    lexicon = Lexicon(acoustic_labels)
    for word in words:
        lexicon.add(word['word'], word['kana'])
    lexicon_fst = lexicon.create_fst(vocab)

    token = Token()
    token_fst = token.create_fst(acoustic_labels)

    fst = pywrapfst.compose(token_fst.arcsort('olabel'), lexicon_fst)
    fst = pywrapfst.determinize(fst)


def test_compose_token_and_lexicon_fst_with_homophones(workdir):
    words = [
        {'word': '愛', 'kana': ['ア', 'イ'], 'eos': False},
        {'word': '藍', 'kana': ['ア', 'イ'], 'eos': True}
    ]
    vocab = get_vocabulary_symbol(workdir, words)

    acoustic_labels = MonophoneLabels(phonemes, kana2phonemes)
    lexicon = Lexicon(acoustic_labels)
    for word in words:
        lexicon.add(word['word'], word['kana'])
    lexicon_fst = lexicon.create_fst(vocab, min_freq=0)

    token = Token()
    token_fst = token.create_fst(acoustic_labels)

    fst = pywrapfst.compose(token_fst.arcsort('olabel'), lexicon_fst)
    with pytest.raises(pywrapfst.FstOpError):
        pywrapfst.determinize(fst)


@pytest.fixture
def words_without_homophones():
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
def words_with_homophones():
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


def test_grammar_create_fst(workdir, words_without_homophones):
    corpus_path = os.path.join(workdir, 'corpus.txt')
    corpus = Corpus()
    for word in words_without_homophones:
        corpus.add(word['word'], end_of_sentence=word['eos'])
    corpus.save(corpus_path)

    vocab_path = os.path.join(workdir, 'vocab.syms')
    vocab = VocabularySymbol()
    vocab.create(vocab_path, corpus_path)

    grammar_path = os.path.join(workdir, 'grammar.fst')
    grammar = Grammar()
    grammar.create_fst(grammar_path, vocab_path, corpus_path)


def test_wfst_decoder_create_fst(workdir, words_without_homophones):
    corpus_path = os.path.join(workdir, 'corpus.txt')
    corpus = Corpus()
    for word in words_without_homophones:
        corpus.add(word['word'], end_of_sentence=word['eos'])
    corpus.save(corpus_path)

    vocab_path = os.path.join(workdir, 'vocab.syms')
    vocab = VocabularySymbol()
    vocab.create(vocab_path, corpus_path)
    vocab.read(vocab_path)

    acoustic_labels = MonophoneLabels(phonemes, kana2phonemes)
    lexicon = Lexicon(acoustic_labels)
    for word in words_without_homophones:
        lexicon.add(word['word'], word['kana'])
    lexicon_fst = lexicon.create_fst(vocab, min_freq=0)

    token = Token()
    token_fst = token.create_fst(acoustic_labels)

    grammar_path = os.path.join(workdir, 'grammar.fst')
    grammar = Grammar()
    grammar_fst = grammar.create_fst(grammar_path, vocab_path, corpus_path)

    wfst_decoder = WFSTDecoder()
    wfst_decoder.create_fst(token_fst, lexicon_fst, grammar_fst)


def test_wfst_decoder_create_fst_with_homophones(
        workdir, words_with_homophones):
    corpus_path = os.path.join(workdir, 'corpus.txt')
    corpus = Corpus()
    for word in words_with_homophones:
        corpus.add(word['word'], end_of_sentence=word['eos'])
    corpus.save(corpus_path)

    vocab_path = os.path.join(workdir, 'vocab.syms')
    vocab = VocabularySymbol()
    vocab.create(vocab_path, corpus_path)
    vocab.read(vocab_path)

    acoustic_labels = MonophoneLabels(phonemes, kana2phonemes)

    lexicon = Lexicon(acoustic_labels)
    for word in words_with_homophones:
        lexicon.add(word['word'], word['kana'])
    lexicon_fst = lexicon.create_fst(vocab, min_freq=0)

    token = Token()
    token_fst = token.create_fst(acoustic_labels)

    grammar_path = os.path.join(workdir, 'grammar.fst')
    grammar = Grammar()
    grammar_fst = grammar.create_fst(grammar_path, vocab_path, corpus_path)

    wfst_decoder = WFSTDecoder()
    wfst_decoder.create_fst(token_fst, lexicon_fst, grammar_fst)


def test_wfst_decoder_decode(workdir, words_with_homophones):
    corpus_path = os.path.join(workdir, 'corpus.txt')
    corpus = Corpus()
    for word in words_with_homophones:
        corpus.add(word['word'], end_of_sentence=word['eos'])
    corpus.save(corpus_path)

    vocab_path = os.path.join(workdir, 'vocab.syms')
    vocab = VocabularySymbol()
    vocab.create(vocab_path, corpus_path)
    vocab.read(vocab_path)

    acoustic_labels = MonophoneLabels(phonemes, kana2phonemes)
    lexicon = Lexicon(acoustic_labels)
    for word in words_with_homophones:
        lexicon.add(word['word'], word['kana'])
    lexicon_fst = lexicon.create_fst(vocab, min_freq=0)

    token = Token()
    token_fst = token.create_fst(acoustic_labels)

    grammar_path = os.path.join(workdir, 'grammar.fst')
    grammar = Grammar()
    grammar_fst = grammar.create_fst(grammar_path, vocab_path, corpus_path)

    wfst_decoder = WFSTDecoder()
    wfst_decoder.create_fst(token_fst, lexicon_fst, grammar_fst)

    blank_id = acoustic_labels.get_blank_id()
    a = acoustic_labels.get_label_id('a')
    i = acoustic_labels.get_label_id('i')
    d = acoustic_labels.get_label_id('d')
    e = acoustic_labels.get_label_id('e')
    s = acoustic_labels.get_label_id('s')
    o = acoustic_labels.get_label_id('o')
    m = acoustic_labels.get_label_id('m')
    r = acoustic_labels.get_label_id('r')
    u = acoustic_labels.get_label_id('u')
    frame_labels = [blank_id, blank_id, a, a, i, i, i, d, e, blank_id,
                    s, s, o, o, o, m, e, r, r, u]
    got = wfst_decoder.decode(frame_labels, vocab)
    assert got == '藍で染める'


def test_wfst_decoder_epsilon_transition():
    acoustic_labels = MonophoneLabels(phonemes, kana2phonemes)

    fst_compiler = _FstCompiler()
    eps = acoustic_labels.get_epsilon_id()
    a = acoustic_labels.get_label_id('a')
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
    wfst_decoder.epsilon_transition(paths, acoustic_labels, frame_index)
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
    acoustic_labels = MonophoneLabels(phonemes, kana2phonemes)

    fst_compiler = _FstCompiler()
    eps = acoustic_labels.get_epsilon_id()
    blank = acoustic_labels.get_blank_id()
    a = acoustic_labels.get_label_id('a')
    i = acoustic_labels.get_label_id('i')
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
