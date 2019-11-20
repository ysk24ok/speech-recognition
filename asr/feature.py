import json

import librosa
import numpy
import scipy


eps = numpy.finfo(numpy.float32).eps


# TODO: Use Data Class instead of namedtuple, but it requires python >= 3.7
class FeatureParams(object):

    def __init__(self, fft_size, frame_length, hop_length, feature_size):
        self.fft_size = fft_size
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.feature_size = feature_size

    def save(self, path):
        params = {
            "fft_size": self.fft_size,
            "frame_length": self.frame_length,
            "hop_length": self.hop_length,
            "feature_size": self.feature_size
        }
        with open(path, 'w') as f:
            json.dump(params, f)

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return FeatureParams(**data)


def pre_emphasis_filter(samples, p=0.97):
    """
    Args:
        samples (numpy.ndarray): audio samples, shape=(n,)
    Returns:
        numpy.ndarray: samples pre-emphasis filter is applied
    """
    return scipy.signal.lfilter([1.0, -p], 1, samples)


def log_mel_spectrum(samples, sampling_rate, fft_size=2048,
                     frame_length_in_sec=0.025, hop_length_in_sec=0.01,
                     filter_bank=24):
    """
    Args:
        samples (numpy.ndarray): audio samples, shape=(n,)
        sampling_rate (int): sampling rate
        fft_size (int): FFT window size, default is 2048
        frame_length_in_sec (float): frame length in seconds
        hop_length_in_sec (float): frame interval in seconds
        filter_bank (int): number of filters
    Returns:
        numpy.ndarray: log-mel spectrum, shape=(number of frames, filter bank)
    """
    power_spectrum = numpy.abs(librosa.core.stft(
        pre_emphasis_filter(samples),
        n_fft=fft_size,
        hop_length=int(sampling_rate * hop_length_in_sec),
        win_length=int(sampling_rate * frame_length_in_sec)
    ))
    mel_filter_bank = librosa.filters.mel(
        sr=sampling_rate,
        n_fft=fft_size,
        n_mels=filter_bank
    )
    # Add epsilon to prevent division by zero in numpy.log
    return numpy.log(mel_filter_bank @ power_spectrum + eps).T


def extract_feature_from_wavfile(wav_filepath, feature_params,
                                 feature_extraction='mel_spectrum'):
    """Extract audio feature from Wave file
    Args:
        wav_filepath (str): Wave file path
        feature_params (FeatureParams): FeatureParams object
        feature_extraction (str): feature extraction method
    Returns:
        numpy.ndarray: shape=(number of frames, feature size)
    """
    rate, data = scipy.io.wavfile.read(wav_filepath)
    if feature_extraction == 'mel_spectrum':
        return log_mel_spectrum(
            data, rate, feature_params.fft_size, feature_params.frame_length,
            feature_params.hop_length, feature_params.feature_size)
    else:
        raise ValueError('feature_extraction: {} is not supported.'.format(
            feature_extraction))
