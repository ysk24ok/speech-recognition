import pytest

from asr.acoustic_labels import MonophoneLabels


@pytest.fixture
def monophone_labels():
    phonemes = ['a', 'k']
    kana2phonemes = {'ア': 'a', 'カ': 'k a'}
    return MonophoneLabels(phonemes, kana2phonemes)


def test_monophone_labels_get_epsilon_id(monophone_labels):
    assert monophone_labels.get_epsilon_id() == 0


def test_monophone_labels_get_blank_id(monophone_labels):
    assert monophone_labels.get_blank_id() == 1


def test_monophone_labels_get_label_id(monophone_labels):
    assert monophone_labels.get_label_id('a') == 2
    assert monophone_labels.get_label_id('k') == 3


def test_monophone_labels_get_auxiliary_label_id(monophone_labels):
    monophone_labels.set_auxiliary_label('#0')
    monophone_labels.set_auxiliary_label('#1')
    assert monophone_labels.get_auxiliary_label_id('#0') == 4
    assert monophone_labels.get_auxiliary_label_id('#1') == 5


def test_monophone_labels_get_label(monophone_labels):
    monophone_labels.set_auxiliary_label('#0')
    monophone_labels.set_auxiliary_label('#1')
    assert monophone_labels.get_label(0) == '<epsilon>'
    assert monophone_labels.get_label(1) == '<blank>'
    assert monophone_labels.get_label(2) == 'a'
    assert monophone_labels.get_label(3) == 'k'
    assert monophone_labels.get_label(4) == '#0'
    assert monophone_labels.get_label(5) == '#1'


def test_monophone_labels_get_label_ids(monophone_labels):
    assert monophone_labels.get_label_ids('ア') == [2]
    assert monophone_labels.get_label_ids('カ') == [3, 2]


def test_monophone_labels_get_all_labels(monophone_labels):
    got = monophone_labels.get_all_labels()
    assert isinstance(got, dict)
    assert len(got) == 4
    assert got[0] == '<epsilon>'
    assert got[1] == '<blank>'
    assert got[2] == 'a'
    assert got[3] == 'k'


def test_monophone_labels_get_all_auxiliary_labels(monophone_labels):
    monophone_labels.set_auxiliary_label('#0')
    monophone_labels.set_auxiliary_label('#1')
    got = monophone_labels.get_all_auxiliary_labels()
    assert isinstance(got, dict)
    assert len(got) == 2
    assert got[4] == '#0'
    assert got[5] == '#1'
