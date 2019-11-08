import numpy as np

from ..preprocessings.signal_processing import resample, pad_signal


def test_resample():
    """
    Check that:
    - Undersampled -> Oversampled reconstrutited signal is close enough to the original signal
    - The length is in agreement with the downsampling/oversampling factor
    """
    sampling_frequency_base = 128
    sampling_frequency_target = 64
    sig_properties = {'padding': 0, 'fs': sampling_frequency_base}

    signal_length = 100000 // sampling_frequency_base * sampling_frequency_base
    signals_frequency = 8
    base_sig = np.cos(
        np.arange(0, signal_length) * np.pi * signals_frequency / sampling_frequency_base)
    base_sig = np.expand_dims(base_sig, -1)
    sig, sig_properties = resample(base_sig, signal_properties=sig_properties,
                                   target_frequency=sampling_frequency_target)

    assert sig_properties['fs'] == sampling_frequency_target
    assert len(sig) == round(
        base_sig.shape[0] / (sampling_frequency_base / sampling_frequency_target))
    sig, sig_properties = resample(sig, signal_properties=sig_properties,
                                   target_frequency=sampling_frequency_base)

    assert sig_properties['fs'] == sampling_frequency_base

    assert len(sig) == base_sig.shape[0]
    print(sig)
    print(base_sig)
    assert np.mean((sig - base_sig) ** 2) < 1e-4


def test_padding_base():
    """
    Test that padding adds the right number of steps given the fs.
    """
    padding = 300
    fs = 256
    signal_length, n_channels = int(10e4), 40
    signal = np.abs(np.random.normal(size=(signal_length, n_channels))) + 1
    sig_properties = {'padding': 0, 'fs': fs}
    signal_padded, sig_properties = pad_signal(signal, sig_properties, padding_duration=padding,
                                               value=0)
    assert np.sum(signal_padded == 0) == 2 * fs * padding * n_channels
    if padding > 0:
        assert np.sum(np.abs(signal_padded[fs * padding:-(fs * padding)] - signal)) == 0
    else:
        assert np.sum(signal_padded - signal) == 0
        assert sig_properties['padding'] == padding
        assert sig_properties['fs'] == fs
    if padding > 0:
        assert np.sum(np.abs(signal_padded[:fs * padding] - signal_padded[-fs * padding:])) == 0
        assert np.mean(signal_padded[:fs * padding] == 0) == 1


def test_padding_composition():
    """
    assert that padding(a+b) == padding(a) + padding(b)
    """
    padding_a = 20
    padding_b = 50
    fs = 256
    signal_length, n_channels = int(10e4), 40
    signal = np.abs(np.random.normal(size=(signal_length, n_channels))) + 1
    sig_properties = {'padding': 0, 'fs': fs}
    signal_padded_a, sig_properties_a = pad_signal(signal, sig_properties,
                                                   padding_duration=padding_a,
                                                   value=0)
    signal_padded_a_and_b, sig_properties_a_and_b = pad_signal(signal_padded_a, sig_properties_a,
                                                               padding_duration=padding_b,
                                                               value=0)
    signal_padded_all, sig_properties_all = pad_signal(signal, sig_properties,
                                                       padding_duration=padding_a + padding_b,
                                                       value=0)
    assert np.sum(np.abs(signal_padded_a_and_b - signal_padded_all)) == 0
    assert sig_properties_a_and_b == sig_properties_all
