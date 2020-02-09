import argparse
import os

import torch

from asr import phonemes
from asr.acoustic_model import AcousticModel
from asr.decoder import WFSTDecoder
from asr.feature import extract_feature_from_wavfile, FeatureParams
from asr.label_table import PhonemeTable, VocabularySymbolTable
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

phoneme_table = PhonemeTable()
phoneme_table.add_labels(phonemes)
epsilon_id = phoneme_table.get_epsilon_id()
print('Loading model ...')
model_path = os.path.join(args.workdir, args.model_file)
model = AcousticModel.load(model_path, blank=phoneme_table.get_blank_id())
feature_params_path = os.path.join(args.workdir, args.feature_params_file)
feature_params = FeatureParams.load(feature_params_path)
batch = []
for wav_file in args.wav_files:
    data = extract_feature_from_wavfile(wav_file, feature_params)
    batch.append(torch.from_numpy(data))
output = model.predict(pad_sequence(batch))
for idx, wav_file in enumerate(args.wav_files):
    print('Decoding {} ... '.format(wav_file))
    frame_labels = [int(frame_label) for frame_label in output[:, idx]]
    print('  acoustic labels = {}'.format(' '.join(
        [phoneme_table.get_label(frame_label) for frame_label in frame_labels
         if frame_label != phoneme_table.get_blank_id()]))
    )
    vocabulary_symbol_path = os.path.join(
        args.workdir, args.vocabulary_symbol_file)
    vocab_symbol = VocabularySymbolTable.load_symbol(
        vocabulary_symbol_path)
    decoder_fst_path = os.path.join(args.workdir, args.decoder_fst_file)
    wfst_decoder = WFSTDecoder()
    wfst_decoder.read_fst(decoder_fst_path)
    print('  text = {} '.format(wfst_decoder.decode(
        frame_labels, vocab_symbol, epsilon_id=epsilon_id)))
