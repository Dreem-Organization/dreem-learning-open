import numpy as np
from scipy import interpolate
from scipy.signal import iirfilter, lfilter, filtfilt


def filter_base(signal, axis=-1, fs=250.):
    """ Perform 60 Hz notch filtering using scipy library.

    Parameters
    ----------
    signal : 1D numpy.array
        Array to filter.
    axis: int
        Choose axis where to perform filtering.
    forward_backward : boolean
        Set True if you want a null phase shift filtered signal

    Returns
    -------
        1D numpy.array
            The signal filtered
    """
    b1, a1 = iirfilter(2,
                       [0.4 / (fs / 2.), 18 / (fs / 2.)],
                       btype='bandpass',
                       ftype='butter')

    b2, a2 = iirfilter(3,
                       [58 / (fs / 2.), 62 / (fs / 2.)],
                       btype='bandstop',
                       ftype='butter')
    b3, a3 = iirfilter(3,
                       [48 / (fs / 2.), 52 / (fs / 2.)],
                       btype='bandstop',
                       ftype='butter')
    b4, a4 = iirfilter(1,
                       [62 / (fs / 2.), 63 / (fs / 2.)],
                       btype='bandstop',
                       ftype='bessel')

    a = np.polymul(np.polymul(np.polymul(a1, a2), a3), a4)
    b = np.polymul(np.polymul(np.polymul(b1, b2), b3), b4)
    result = lfilter(b, a, signal, axis)
    return result


def filter(signal, signal_properties):
    signal = filter_base(signal, axis=0, fs=signal_properties['fs'])
    return signal, signal_properties


def iir_bandpass_filter(signal,
                        fs=250.,
                        order=4,
                        frequency_band=[0.4, 18],
                        filter_type='butter',
                        axis=-1,
                        forward_backward=False):
    """ Perform bandpass filtering using scipy library.

    Parameters
    ----------

    signal : 1D numpy.array
        Array to filter.
    fs : float
        Sampling frequency
    order : int
        Order of the filter
    frequency_band : list
        Specify bandpass eg: [0.5, 20] will keep frequencies between 0.5
        and 20 Hz
    filter_type : str
        Choose type of IIR filter: butter, cheby1, cheby2, ellip, bessel
    axis: int
        Choose axis where to perform filtering.
    forward_backward : boolean
        Set True if you want a null phase shift filtered signal

    Returns
    -------

        1D numpy.array
            The signal filtered
    """
    b, a = iirfilter(order,
                     [ff * 2. / fs for ff in frequency_band],
                     btype='bandpass',
                     ftype=filter_type)
    if forward_backward:
        result = filtfilt(b, a, signal, axis)
    else:
        result = lfilter(b, a, signal, axis)
    return result


def filter_cardio(signal, signal_properties):
    signal = iir_bandpass_filter(signal,
                                 fs=signal_properties['fs'],
                                 order=2,
                                 frequency_band=[0.6, 3],
                                 filter_type='bessel',
                                 axis=0,
                                 forward_backward=False)
    return signal, signal_properties


def filter_respiration(signal, signal_properties):
    signal = iir_bandpass_filter(signal,
                                 fs=signal_properties['fs'],
                                 order=2,
                                 frequency_band=[0.1, 0.5],
                                 filter_type='bessel',
                                 axis=0,
                                 forward_backward=False)
    return signal, signal_properties


def resample(signal, signal_properties, target_frequency, interpolation_args={}):
    signal_frequency = signal_properties['fs']
    resampling_ratio = signal_frequency / target_frequency
    x_base = np.arange(0, len(signal))

    interpolator = interpolate.interp1d(x_base, signal, axis=0, bounds_error=False,
                                        fill_value='extrapolate',
                                        **interpolation_args)

    x_interp = np.arange(0, len(signal), resampling_ratio)

    signal_duration = signal.shape[0] / signal_frequency
    resampled_length = round(signal_duration * target_frequency)
    resampled_signal = interpolator(x_interp)
    if len(resampled_signal) < resampled_length:
        padding = np.zeros((resampled_length - len(resampled_signal), signal.shape[-1]))
        resampled_signal = np.concatenate([resampled_signal, padding])

    signal_properties = {'fs': target_frequency, 'padding': signal_properties['padding']}
    return resampled_signal, signal_properties


def weighted_sum(signal, signal_properties, weights=None):
    if weights is None:
        return np.sum(signal, -1, keepdims=True), signal_properties
    else:
        return np.sum(signal * np.array(weights), -1, keepdims=True), signal_properties


def pad_signal(signal, signal_properties, padding_duration, value=0):
    if padding_duration == 0:
        return signal, signal_properties
    else:
        fs = signal_properties['fs']
        padding_array = np.zeros((padding_duration * fs,) + signal.shape[1:]) + value
        signal = [padding_array] + [signal] + [padding_array]
        signal_properties = {'fs': fs, 'padding': signal_properties['padding'] + padding_duration}
        return np.concatenate(signal), signal_properties


signal_processings = {
    'filter': filter,
    'filter_cardio': filter_cardio,
    'filter_respiration': filter_respiration,
    'resample': resample,
    'padding': pad_signal,
    'weighted_sum': weighted_sum
}
