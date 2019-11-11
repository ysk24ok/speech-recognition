import pytest

from asr import decoder


@pytest.fixture
def lexicon():
    phonemes = ['u', 'ky', 'sh', ':']
    kana_phoneme_mapping = {
        'ウ': 'u',
        'キュ': 'ky u',
        'シュ': 'sh u',
        'ー': ':'
    }
    return decoder.Lexicon(phonemes, kana_phoneme_mapping)


def test_lexicon_add(lexicon):
    word = '九州'
    reading = ['キュ', 'ー', 'シュ', 'ー']
    lexicon.add(word, reading)
    expected = [1, 0, 3, 2, 0, 3]
    assert(lexicon.get(word) == expected)
    wrong_reading = ['キュ', 'ウ', 'シュ', 'ウ']
    lexicon.add(word, wrong_reading)
    expected = [1, 0, 3, 2, 0, 3]
    assert(lexicon.get(word) == expected)
