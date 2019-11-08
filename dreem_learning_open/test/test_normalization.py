import time

import numpy as np
import torch

from ..models.modulo_net.normalization import multi_std, multi_mean, spectrogram


def test_mean_torch_numpy():
    """
    Test that the mean computed with torch on multiple axis is the same that the one computed with numpy
    """
    locs = range(10)
    X = []
    for loc in locs:
        X += [np.random.normal(loc=loc, size=(1, 10, 100))]
    X = np.concatenate(X)
    mean_numpy = np.mean(X, (1, 2), keepdims=True)
    mean_torch = multi_mean(torch.from_numpy(X), (1, 2)).numpy()
    assert np.linalg.norm(mean_torch - mean_numpy) < 1e-6


def test_std_torch_numpy():
    """
    Test that the std computed with torch on multiple axis is the same that the one computed with numpy
    """
    scales = range(10)
    X = []
    for scale in scales:
        X += [np.random.normal(scale=scale, size=(1, 10, 100))]
    X = np.concatenate(X)
    std_numpy = np.std(X, (1, 2), keepdims=True)
    std_torch = multi_std(torch.from_numpy(X), (1, 2), unbiased=False).numpy()
    assert np.linalg.norm(std_torch - std_numpy) < 1e-6


def test_spectrogram():
    """
    Test on different sine:
    - that the spectrogram has the right shape
    - that the maximum is produced at the right frequency
    """

    # Generating sine wave at 5,10 and 15 Hz
    sine_frequency = [5, 10, 15]
    signal_length = 1e5
    signal_sampling_frequency = 100
    temporal_context = 5
    window_durations = int(signal_length / signal_sampling_frequency / temporal_context - 2)
    X = []
    for fs in sine_frequency:
        X += [np.expand_dims(
            np.cos(
                2 * np.arange(0, signal_length) * np.pi * fs / signal_sampling_frequency).reshape(5,
                                                                                                  -1),
            1)]
    X = np.concatenate(X, 1).transpose((1, 0, 2))
    X = torch.from_numpy(np.expand_dims(X, 0)).contiguous().float()
    X = spectrogram(X, signal_sampling_frequency)

    # Test that the duration is the signal duration minus 2 seconds (padding)
    assert X.shape[-1] == window_durations
    values, indices = X.max(0)[0].max(2)

    # Assert that the produced spectrogram are coherent with the signal frequency
    for i, fs in enumerate(sine_frequency):
        assert len(torch.unique(indices[i])) == 1
        assert torch.norm(
            (indices[i][0, 0].float() / X.shape[-2] * signal_sampling_frequency / 2) - fs) < 1


def test_cpu_vs_gpu(verbose=False):
    """
    assert that running on gpu is faster than on cpu
    """
    if torch.cuda.is_available():
        batch_size = 16
        window_size = 3000
        fs = 100
        temporal_context = 11
        N_channels = 3
        N_test = 100

        signal = np.random.normal(size=(batch_size, temporal_context, N_channels, window_size))
        signal_cpu = torch.from_numpy(signal).float()
        signal_gpu = torch.from_numpy(signal).float().to('cuda')

        # Test on standard deviation computation
        begin_time = time.time()
        for i in range(N_test):
            multi_std(signal_gpu, axis=(0, 1, -1))
        end_time = time.time()
        std_gpu_duration = (end_time - begin_time)

        begin_time = time.time()
        for i in range(N_test):
            multi_std(signal_cpu, axis=(0, 1, -1))
        end_time = time.time()
        std_cpu_duration = (end_time - begin_time)
        assert std_cpu_duration / std_gpu_duration > 1
        if verbose:
            print('gpu is ', round(std_cpu_duration / std_gpu_duration, 1),
                  ' times faster than cpu on std')

        # Test on standard deviation computation
        begin_time = time.time()
        for i in range(N_test):
            spectrogram(signal_gpu, fs=fs)
        end_time = time.time()
        spectro_gpu_duration = (end_time - begin_time)

        begin_time = time.time()
        for i in range(N_test):
            spectrogram(signal_cpu, fs=fs)
        end_time = time.time()
        spectro_cpu_duration = (end_time - begin_time)
        assert spectro_cpu_duration / spectro_gpu_duration > 1
        if verbose:
            print('gpu is ', round(spectro_cpu_duration / spectro_gpu_duration, 1),
                  'times faster than cpu on spectro')

    else:
        pass
