import os
import shutil

import pytest

from asr.label_table import (
    PhonemeTable,
    VocabularyTable,
    VocabularySymbolTable
)
from asr.decoder import Corpus


@pytest.fixture
def workdir():
    dirpath = '/tmp/work'
    os.mkdir(dirpath)
    yield dirpath
    shutil.rmtree(dirpath)


def test_phoneme_table_get_epsilon_id():
    phoneme_table = PhonemeTable()
    assert phoneme_table.get_epsilon_id() == 0
    assert phoneme_table.get_label_id('<epsilon>') == 0
    assert phoneme_table.get_label(0) == '<epsilon>'


def test_phoneme_table_get_blank_id():
    phoneme_table = PhonemeTable()
    assert phoneme_table.get_blank_id() == 1
    assert phoneme_table.get_label_id('<blank>') == 1
    assert phoneme_table.get_label(1) == '<blank>'


def test_phoneme_table_add_label():
    phoneme_table = PhonemeTable()
    phoneme_table.add_label('a')
    assert phoneme_table.num_labels() == 3
    assert phoneme_table.get_label_id('a') == 2
    assert phoneme_table.get_label(2) == 'a'


def test_phoneme_table_add_labels():
    phoneme_table = PhonemeTable()
    phoneme_table.add_labels(['a', 'i'])
    assert phoneme_table.num_labels() == 4
    assert phoneme_table.get_label_id('a') == 2
    assert phoneme_table.get_label(2) == 'a'
    assert phoneme_table.get_label_id('i') == 3
    assert phoneme_table.get_label(3) == 'i'


def test_phoneme_table_get_auxiliary_label_id():
    phoneme_table = PhonemeTable()
    phoneme_table.add_label('a')
    phoneme_table.set_auxiliary_label('#0')
    phoneme_table.set_auxiliary_label('#1')
    assert phoneme_table.get_label_id('a') == 2
    assert phoneme_table.get_auxiliary_label_id('#0') == 3
    assert phoneme_table.get_auxiliary_label_id('#1') == 4


def test_phoneme_table_get_all_labels():
    phoneme_table = PhonemeTable()
    phoneme_table.add_label('a')
    got = phoneme_table.get_all_labels()
    assert isinstance(got, dict)
    assert len(got) == 3
    assert got[0] == '<epsilon>'
    assert got[1] == '<blank>'
    assert got[2] == 'a'


def test_phoneme_table_get_all_auxiliary_labels():
    phoneme_table = PhonemeTable()
    phoneme_table.add_label('a')
    phoneme_table.set_auxiliary_label('#0')
    phoneme_table.set_auxiliary_label('#1')
    got = phoneme_table.get_all_auxiliary_labels()
    assert isinstance(got, dict)
    assert len(got) == 2
    assert got[3] == '#0'
    assert got[4] == '#1'


def test_vocabulary_table_get_pad_id():
    vocab_table = VocabularyTable()
    assert vocab_table.get_pad_id() == 0
    assert vocab_table.get_label_id('<pad>') == 0
    assert vocab_table.get_label(0) == '<pad>'


def test_vocabulary_table_get_unk_id():
    vocab_table = VocabularyTable()
    assert vocab_table.get_unk_id() == 1
    assert vocab_table.get_label_id('<unk>') == 1
    assert vocab_table.get_label(1) == '<unk>'


def test_vocabulary_table_get_bos_id():
    vocab_table = VocabularyTable()
    assert vocab_table.get_bos_id() == 2
    assert vocab_table.get_label_id('<bos>') == 2
    assert vocab_table.get_label(2) == '<bos>'


def test_vocabulary_table_get_eos_id():
    vocab_table = VocabularyTable()
    assert vocab_table.get_eos_id() == 3
    assert vocab_table.get_label_id('<eos>') == 3
    assert vocab_table.get_label(3) == '<eos>'


def test_vocabulary_table_add_label():
    # when min_freq is 1
    vocab_table = VocabularyTable()
    vocab_table.add_label('私')
    assert vocab_table.num_labels() == 5
    assert vocab_table.get_label_id('私') == 4
    assert vocab_table.get_label(4) == '私'
    # when min_freq is 2 or more
    vocab_table = VocabularyTable(min_freq=2)
    vocab_table.add_label('私')
    assert vocab_table.num_labels() == 4
    assert vocab_table.get_label_id('私') == vocab_table.get_unk_id()
    vocab_table.add_label('私')
    assert vocab_table.num_labels() == 5
    assert vocab_table.get_label_id('私') == 4


def test_vocabulary_table_save_and_load(workdir):
    vocab_path = os.path.join(workdir, 'vocab.txt')
    vocab_table = VocabularyTable()
    vocab_table.add_label('私')
    vocab_table.add_label('あなた')
    vocab_table.add_label('あなた')
    vocab_table.save(vocab_path)

    vocab_loaded = VocabularyTable.load(vocab_path)
    assert vocab_loaded.get_label_id('<pad>') == 0
    assert vocab_loaded.get_label(0) == '<pad>'
    assert vocab_loaded.get_label_id('<unk>') == 1
    assert vocab_loaded.get_label(1) == '<unk>'
    assert vocab_loaded.get_label_id('<bos>') == 2
    assert vocab_loaded.get_label(2) == '<bos>'
    assert vocab_loaded.get_label_id('<eos>') == 3
    assert vocab_loaded.get_label(3) == '<eos>'
    assert vocab_loaded.get_label_id('私') == 4
    assert vocab_loaded.get_label(4) == '私'
    assert vocab_loaded.get_label_id('あなた') == 5
    assert vocab_loaded.get_label(5) == 'あなた'

    vocab_loaded = VocabularyTable.load(vocab_path, min_freq=2)
    assert vocab_loaded.get_label_id('あなた') == 4
    assert vocab_loaded.get_label(4) == 'あなた'
    assert vocab_loaded.get_label_id('私') == vocab_loaded.get_unk_id()

    vocab_loaded = VocabularyTable.load(vocab_path, min_freq=3)
    assert vocab_loaded.get_label_id('あなた') == vocab_loaded.get_unk_id()
    assert vocab_loaded.get_label_id('私') == vocab_loaded.get_unk_id()


def test_vocabulary_table_create_and_load_symbol(workdir):
    corpus_path = os.path.join(workdir, 'corpus.txt')
    corpus = Corpus()
    corpus.add('ボール', end_of_sentence=False)
    corpus.add('を', end_of_sentence=False)
    corpus.add('投げる', end_of_sentence=True)
    corpus.save(corpus_path)

    vocab_path = os.path.join(workdir, 'vocab.syms')
    VocabularySymbolTable.create_symbol(vocab_path, corpus_path)
    vocab_table = VocabularySymbolTable.load_symbol(vocab_path)
    assert vocab_table.get_label_id('<epsilon>') == 0
    assert vocab_table.get_label(0) == '<epsilon>'
    assert vocab_table.get_label_id('ボール') == 1
    assert vocab_table.get_label(1) == 'ボール'
    assert vocab_table.get_label_id('を') == 2
    assert vocab_table.get_label(2) == 'を'
    assert vocab_table.get_label_id('投げる') == 3
    assert vocab_table.get_label(3) == '投げる'
    assert vocab_table.get_label_id('<unk>') == 4
    assert vocab_table.get_label(4) == '<unk>'
