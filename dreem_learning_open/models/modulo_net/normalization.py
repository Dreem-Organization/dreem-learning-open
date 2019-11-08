"""
Implementation of normalization operators. Normalization operators ensure the consistency fo the data within the model.
"""

import copy

import numpy as np
import torch
import tqdm


def clip(x, value):
    return torch.clamp(x, -value, value)


def affine(x, gain, bias=0):
    return x * gain + bias


def torch_spectrogram(signal, n_fft=None, hop=None, window=np.hanning):
    window = window(n_fft) / window(n_fft).sum()
    window = torch.from_numpy(window).to(signal.device).float()
    stft = torch.stft(signal, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=window, onesided=True, center=False, normalized=False)
    stft = (stft ** 2).sum(-1)
    return stft


def spectrogram(x, fs, window_duration=2, window_overlap=1, logpower=True, clamp=1e-20):
    fs = fs
    ws, hop = 2 ** int(np.log2(fs * window_duration) + 1), fs * window_overlap
    batch_size, temporal_context, n_channels, signal_length = x.shape
    x = x.view(batch_size * temporal_context, n_channels, signal_length).view(-1, signal_length)
    x = torch_spectrogram(x, hop=hop, n_fft=ws).float()
    x = torch.clamp(x, clamp, 1e32)
    if logpower:
        x = torch.log10(x)
    frequency_features, spectro_length = x.shape[1], x.shape[2]
    x = x.view(batch_size, temporal_context, n_channels, frequency_features, spectro_length)
    return x


def clip_and_scale(x, min_value, max_value):
    return x.clamp(min=min_value, max=max_value) / max([abs(min_value), abs(max_value)])


def multi_std(x, axis, unbiased=True):
    base_shape = list(x.shape)
    for ax in axis:
        x = x.unsqueeze(-1)
        x = x.transpose(ax, -1)
        base_shape[ax] = 1
    new_shape = tuple(base_shape + [-1])
    x = x.contiguous().view(new_shape)
    return x.std(-1, unbiased=unbiased)


def multi_mean(x, axis):
    for ax in axis:
        x = x.mean(ax, keepdim=True)
    return x


def standardize(x, mu=None, sigma=None, eps=1e-15):
    if mu is None or sigma is None:
        raise ValueError('mu and sigma have to be precomputed')
    x = (x - mu) / (sigma + eps)
    return x


normalizers = {
    'clip': clip,
    'affine': affine,
    'standardization': standardize,
    'standardize': standardize,
    'spectrogram': spectrogram,
    'clip_and_scale': clip_and_scale
}

features_normalizer = {
    'clip_and_scale': clip_and_scale
}


def normalize(sample, normalization_parameters):
    for group_normalization in normalization_parameters:
        group_name = group_normalization['name']
        for normalization in group_normalization['normalization']:
            sample[group_name] = normalizers[normalization['type']](
                sample[group_name], **normalization['args'])
    return sample


def normalize_features(feature, normalization_parameters):
    for feature_normalization in normalization_parameters:
        feature_name = feature_normalization['name']
        for normalization in feature_normalization['normalization']:
            feature[feature_name] = normalizers[normalization['type']](
                feature[feature_name], **normalization['args'])
    return feature


def initialize_standardization_parameters(dataset, normalization_parameters):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalization_parameters = copy.deepcopy(normalization_parameters)
    vars = {'features': 'features', 'signals': 'groups'}
    vars_axis = {'features': (0, 1), 'signals': (0, 1, -1)}
    for var_type, var_name in vars.items():

        for group_normalization in normalization_parameters[var_type]:
            for normalization in group_normalization['normalization']:
                if (normalization['type'] == 'standardize' or
                        normalization['type'] == 'standardization'):
                    if 'mu' not in normalization['args'] or 'sigma' not in normalization['args']:
                        normalization['compute_parameters'] = True
                        normalization['n_updates'] = 0
                    else:
                        normalization['compute_parameters'] = False

    for record in tqdm.tqdm(dataset.records):
        for sample in dataset.get_record(record, 32, mode='eval'):
            for var_type, var_name in vars.items():
                for group_name in sample[var_name]:
                    sample[var_name][group_name] = sample[var_name][group_name].to(device,
                                                                                   non_blocking=True)

                for group_normalization in normalization_parameters[var_type]:
                    group_name = group_normalization['name']
                    for normalization in group_normalization['normalization']:
                        if 'compute_parameters' in normalization:
                            normalize_group = normalization['compute_parameters']
                        else:
                            normalize_group = False
                        if normalize_group:
                            axis = vars_axis[var_type]
                            N = normalization['n_updates']
                            if normalization['n_updates'] == 0:
                                normalization['args']['mu'] = multi_mean(
                                    sample[var_name][group_name], axis=axis)

                                normalization['args']['sigma'] = multi_std(
                                    sample[var_name][group_name], axis=axis)

                            else:

                                normalization['args']['mu'] = normalization['args']['mu'] * N / (
                                        N + 1) + 1 / (
                                                                      N + 1) * multi_mean(
                                    sample[var_name][group_name], axis=axis)
                                normalization['args']['sigma'] = normalization['args'][
                                                                     'sigma'] * N / (N + 1) + 1 / (
                                                                         N + 1) * multi_std(
                                    sample[var_name][group_name], axis=axis)
                            normalization['n_updates'] += 1
                        else:
                            sample[var_name][group_name] = normalizers[normalization['type']](
                                sample[var_name][group_name],
                                **normalization['args'])

    for var_type, var_name in vars.items():
        for group_normalization in normalization_parameters[var_type]:
            for normalization in group_normalization['normalization']:
                try:
                    del normalization['compute_parameters']
                    del normalization['n_updates']
                except KeyError:
                    pass

    return normalization_parameters
