import argparse

import torch

from asr.acoustic_model import AcousticModel
from asr import phonemes
from asr.feature import extract_feature_from_wavfile, FeatureParams
from asr.utils import pad_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--feature-params-path', type=str,
                    default='feature_params.json',
                    help='Feature params path to load')
parser.add_argument('--model-path', type=str, default='model.bin',
                    help='model path to load')
parser.add_argument('wav_files', nargs='*', help='wave files to recognize')
args = parser.parse_args()

print('Loading model ...')
feature_params = FeatureParams.load(args.feature_params_path)
model = AcousticModel.load(args.model_path)
print('Predicting ...')
batch = []
for wav_file in args.wav_files:
    data = extract_feature_from_wavfile(wav_file, feature_params)
    batch.append(torch.from_numpy(data))
output = model.predict(pad_sequence(batch))

predicted_labels = []
for v in output[:, 0]:
    if v == 0:
        continue
    predicted_labels.append(phonemes[v])
print(' '.join(predicted_labels))
