import json

import numpy as np
from scipy.signal import stft
from scipy.stats import entropy


def index_window(signal, signal_properties, increment_duration=30, padding_duration=None):
    signal_frequency = signal_properties['fs']
    index_window = np.arange(0, len(signal) // (increment_duration * signal_frequency))
    if padding_duration is not None:
        padding_pre = [0] * (padding_duration // increment_duration)
        padding_post = [index_window[-1]] * (padding_duration // increment_duration)
        return np.expand_dims(np.concatenate([padding_pre, index_window, padding_post]), -1) / 1200
    else:
        return np.expand_dims(index_window, -1) / 1200


def cyclic_index_window(signal, signal_properties, increment_duration=30,
                        cycles_length=[10, 30, 60, 90, 120],
                        padding_duration=None):
    signal_frequency = signal_properties['fs']
    index_window = np.arange(0, len(signal) // (increment_duration * signal_frequency))
    result = []
    if padding_duration is None:
        padding_duration = 0
    padding_pre = [0] * (padding_duration // increment_duration)
    padding_post = [index_window[-1]] * (padding_duration // increment_duration)
    for cycle_length in cycles_length:
        result += [np.cos(
            np.pi * np.expand_dims(np.concatenate([padding_pre, index_window, padding_post]),
                                   -1) / cycle_length)]

    return np.concatenate(result, -1)


def fft(signal, signal_properties, padding_duration=0, increment_duration=30):
    signal_frequency = signal_properties['fs']
    signal_freq = signal.reshape((-1, increment_duration * signal_frequency))
    signal_freq = np.fft.rfft(signal_freq, n=increment_duration * signal_frequency, axis=1)
    signal_freq = np.abs(signal_freq)
    x = signal_freq

    if padding_duration is None:
        padding_duration = 0
    padding_pre = np.zeros((padding_duration // increment_duration, x.shape[1]))
    padding_post = np.zeros((padding_duration // increment_duration, x.shape[1]))
    x = np.concatenate([padding_pre, x, padding_post], axis=0)
    return x


numpy_statistics = {
    'mean': np.mean,
    'max': np.max,
    'median': np.median,
    'min': np.min,
    'std': np.std
}


def spectral_power(signal, signal_properties, frequency_interval=None, stft_duration=5,
                   stft_overlap=3.5, epoch_duration=30,
                   window='hamming', statistics=None, padding_duration=0):
    N_channels = signal.shape[1]
    fs = signal_properties['fs']

    nperseg, noverlap, nperepoch = int(stft_duration * fs), int(stft_overlap * fs), int(
        epoch_duration * fs)

    N_epoch = signal.shape[0] // nperepoch
    assert N_epoch == signal.shape[0] / nperepoch

    signal = signal.reshape((N_epoch, nperepoch, N_channels)).transpose(
        (0, 2, 1))  # N_epoch, N_channels , nperepoch
    freq, _, signal_stft = stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    signal_stft = np.abs(signal_stft)
    if frequency_interval is not None:
        PSD = []
        for interval in frequency_interval:
            idx_to_sum = np.where(np.logical_and(freq >= interval[0], freq <= interval[1]))[0]
            if len(idx_to_sum) > 0:
                PSD += [np.sum(signal_stft[:, :, idx_to_sum, :], axis=2, keepdims=True)]
        PSD = np.concatenate(PSD, 2)
        if statistics is not None:
            PSD_statistics = []
            for statistic in statistics:
                PSD_statistics += [numpy_statistics[statistic](PSD, axis=3, keepdims=True)]

            PSD_statistics = np.concatenate(PSD_statistics, 3)
            PSD = np.concatenate([PSD, PSD_statistics], axis=3)

    return PSD.reshape((N_epoch, -1))


def shannon_entropy(signal, signal_properties, epoch_duration=30, padding_duration=0):
    fs = signal_properties['fs']
    nperepoch = int(epoch_duration * fs)
    N_epoch = signal.shape[0] // nperepoch
    assert N_epoch == signal.shape[0] / nperepoch
    signal = signal.reshape((N_epoch, nperepoch, -1))
    signal = np.transpose(signal, (1, 0, 2))
    res = entropy(np.abs(signal))
    return res


def signal_statistics(signal, signal_properties, statistics, epoch_duration=30, window_duration=5,
                      padding_duration=0):
    fs = signal_properties['fs']
    nperepoch, nperwindow = int(epoch_duration * fs), int(window_duration * fs)
    N_epoch = signal.shape[0] // nperepoch
    N_channel = signal.shape[1]
    assert N_epoch == signal.shape[0] / nperepoch
    assert epoch_duration // window_duration == epoch_duration / window_duration
    window_per_epoch = epoch_duration // window_duration
    signal = np.abs(signal.reshape((-1, window_per_epoch, nperwindow, N_channel)))

    result = []
    for statistic in statistics:
        result += [numpy_statistics[statistic](signal, axis=2, keepdims=True)]
    return np.concatenate(result, 2).reshape((N_epoch, -1))


epoch_features = {
    'index_window': index_window,
    'cycle_index_window': cyclic_index_window,
    'fft': fft,
    'signal_statistics': signal_statistics,
    'shannon_entropy': shannon_entropy,
    'spectral_power': spectral_power
}
