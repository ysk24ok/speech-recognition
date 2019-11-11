from asr.feature import log_mel_spectrum

import numpy


def test_log_mel_spectrum_default_value():
    # 16-bit PCM
    size = 1000
    data = numpy.random.randint(-32768, 32767, size=size)
    sampling_rate = 8000
    hop_length = sampling_rate * 0.01
    got = log_mel_spectrum(data, sampling_rate)
    assert got.shape == (int(size / hop_length) + 1, 24)
