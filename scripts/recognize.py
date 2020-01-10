import argparse
import os

import torch

from asr import phonemes, kana2phonemes
from asr.acoustic_labels import MonophoneLabels
from asr.acoustic_model import AcousticModel
from asr.decoder import WFSTDecoder, VocabularySymbol
from asr.feature import extract_feature_from_wavfile, FeatureParams
from asr.utils import pad_sequence


parser = argparse.ArgumentParser()
parser.add_argument('workdir', help='Directory path where files are saved')
parser.add_argument('wav_files', nargs='*', help='wave files to recognize')
parser.add_argument('--vocabulary-symbol-file', type=str, default='vocab.syms',
                    help='Vocabulary symbol file name')
parser.add_argument('--feature-params-file', type=str,
                    default='feature_params.json',
                    help='Feature params file name')
parser.add_argument('--model-file', type=str, default='model.bin',
                    help='model file name')
parser.add_argument('--decoder-fst-file', type=str, default='decoder.fst',
                    help='Decoder FST file name')
args = parser.parse_args()

print('Loading model ...')
acoustic_labels = MonophoneLabels(phonemes, kana2phonemes)
model_path = os.path.join(args.workdir, args.model_file)
model = AcousticModel.load(model_path, blank=acoustic_labels.get_blank_id())
print('Predicting acoustic labels ...')
feature_params_path = os.path.join(args.workdir, args.feature_params_file)
feature_params = FeatureParams.load(feature_params_path)
batch = []
for wav_file in args.wav_files:
    data = extract_feature_from_wavfile(wav_file, feature_params)
    batch.append(torch.from_numpy(data))
output = model.predict(pad_sequence(batch))
frame_labels = [int(frame_label) for frame_label in output[:, 0]]
print(' '.join(
    [acoustic_labels.get_label(frame_label) for frame_label in frame_labels
     if frame_label != acoustic_labels.get_blank_id()]
))
print('Decoding ...')
vocabulary_symbol_path = os.path.join(
    args.workdir, args.vocabulary_symbol_file)
vocab_symbol = VocabularySymbol()
vocab_symbol.read(vocabulary_symbol_path)
decoder_fst_path = os.path.join(args.workdir, args.decoder_fst_file)
wfst_decoder = WFSTDecoder()
wfst_decoder.read_fst(decoder_fst_path)
print(wfst_decoder.decode(frame_labels, vocab_symbol,
                          epsilon_id=acoustic_labels.get_epsilon_id()))
