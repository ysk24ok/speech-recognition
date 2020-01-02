import argparse
import os

from asr import phonemes, kana2phonemes
from asr.acoustic_labels import MonophoneLabels
from asr.decoder import (
    Grammar,
    Lexicon,
    Token,
    VocabularySymbol,
    WFSTDecoder
)


parser = argparse.ArgumentParser()
parser.add_argument('workdir',
                    help='Directory path where files are loaded and saved')
parser.add_argument('--vocabulary-symbol-file', type=str, default='vocab.syms',
                    help='Vocabulary symbol file name')
parser.add_argument('--corpus-file', type=str, default='corpus.txt',
                    help='corpus file name')
parser.add_argument('--lexicon-file', type=str, default='lexicon.txt',
                    help='lexicon file name')
parser.add_argument('--token-fst-file', type=str, default='token.fst',
                    help='Token FST file name')
parser.add_argument('--lexicon-fst-file', type=str, default='lexicon.fst',
                    help='Lexicon FST file name')
parser.add_argument('--grammar-fst-file', type=str, default='grammar.fst',
                    help='Grammar FST file name')
parser.add_argument('--decoder-fst-file', type=str, default='decoder.fst',
                    help='Decoder FST file name')
args = parser.parse_args()
print('Creating vocabulary symbol ...')
corpus_path = os.path.join(args.workdir, args.corpus_file)
vocabulary_symbol_path = os.path.join(
    args.workdir, args.vocabulary_symbol_file)
vocab_symbol = VocabularySymbol()
vocab_symbol.create(vocabulary_symbol_path, corpus_path)
vocab_symbol.read(vocabulary_symbol_path)
print('Creating lexicon FST ...')
lexicon_path = os.path.join(args.workdir, args.lexicon_file)
lexicon_fst_filepath = os.path.join(args.workdir, args.lexicon_fst_file)
acoustic_labels = MonophoneLabels(phonemes, kana2phonemes)
lexicon = Lexicon(acoustic_labels)
lexicon.load(lexicon_path)
lexicon_fst = lexicon.create_fst(vocab_symbol)
lexicon_fst.write(lexicon_fst_filepath)
print('Creating token FST ...')
token_fst_path = os.path.join(args.workdir, args.token_fst_file)
token = Token()
token_fst = token.create_fst(acoustic_labels)
token_fst.write(token_fst_path)
print('Creating grammar FST ...')
grammar_fst_path = os.path.join(args.workdir, args.grammar_fst_file)
grammar = Grammar()
grammar_fst = grammar.create_fst(
    grammar_fst_path, vocabulary_symbol_path, corpus_path)
print('Creating decoder ...')
decoder_fst_path = os.path.join(args.workdir, args.decoder_fst_file)
wfst_decoder = WFSTDecoder()
wfst_decoder.create_fst(token_fst, lexicon_fst, grammar_fst)
wfst_decoder.write_fst(decoder_fst_path)
