import numpy as np
import torch
from torch import nn

from dreem_learning_open.models.modulo_net.epochs_encoder.epoch_encoder import EpochEncoder
from dreem_learning_open.models.modulo_net.modules.bn_lstm import LSTM, BNLSTMCell
from torch.nn import GRU

def tri_filter_shape(ndim, nfilter):
    """
    Triangular frequency filter bank used to project the initial freuqncy bank on a smaller and smoother frequency banck.
    from https://github.com/pquochuy/SeqSleepNet
    ndim (int): size of the initial filter bank
    nfilter (int): number of triangular filters to use
    """
    f = np.arange(ndim)
    f_high = f[-1]
    f_low = f[0]
    H = np.zeros((nfilter, ndim))

    M = f_low + np.arange(nfilter + 2) * (f_high - f_low) / (nfilter + 1)
    for m in range(nfilter):
        k = np.logical_and(f >= M[m], f <= M[m + 1])  # up-slope
        H[m][k] = 2 * (f[k] - M[m]) / ((M[m + 2] - M[m]) * (M[m + 1] - M[m]))
        k = np.logical_and(f >= M[m + 1], f <= M[m + 2])  # down-slope
        H[m][k] = 2 * (M[m + 2] - f[k]) / ((M[m + 2] - M[m]) * (M[m + 2] - M[m + 1]))

    H = np.transpose(H)
    H.astype(np.float32)
    return H


def lin_tri_filter_shape(nfilt=20, nfft=129, samplerate=100, lowfreq=0, highfreq=50):
    """Compute a linear-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    # lowmel = self.hz2mel(lowfreq)
    # highmel = self.hz2mel(highfreq)
    # melpoints = np.linspace(lowmel,highmel,nfilt+2)
    hzpoints = np.linspace(lowfreq, highfreq, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft + 1) * hzpoints / samplerate)
    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    fbank = np.transpose(fbank)
    fbank.astype(np.float32)
    return fbank


class FilterBankLayer(nn.Module):
    """
    Layer which implement a differentiable filter bank similarly to https://arxiv.org/abs/1809.10932
    Each filter is non negative, bandwidth limited and the filters are ordered.
    """

    def __init__(self, input_dim=129, filter_dim=20, n_channels=1):
        """

        input_dim (int): size of the original filter bank
        filter_dim (int): size of the fitted filter bank
        n_channels (int): number of distinct signals (i.e. number of channels in the time-frequency representation)
        """
        super().__init__()
        self.input_dim, self.filter_dim = input_dim, filter_dim
        w = lin_tri_filter_shape(filter_dim, (input_dim - 1)* 2 )
        w = np.expand_dims(w, 0)
        W = np.repeat(w, repeats=n_channels, axis=0)
        self.bn = nn.BatchNorm1d(num_features=filter_dim)

        self.triangular_filter_bank = torch.nn.Parameter(data=torch.Tensor(W),
                                                         requires_grad=False)
        filter_weights = torch.nn.init.normal_(torch.Tensor(*W.shape))
        self.filter_weights = torch.nn.Parameter(data=filter_weights,
                                                 requires_grad=True)
        self.register_parameter('filter_weights', self.filter_weights)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """

        x (tensor : batch_size,n_channels,filter_bank_size,time) :
                n_channels : number of signal (for instance EEG,EOG, EMG channels)
                filter_bank_size : frequency resolution of the initial time-frequency representation
                time: number of window/timesteps in the time-frequency representation
        returns (tensor: batch_size, n_channels, self.filter_dim, time)
        """
        batch_size, n_channels, input_filter_dim, length = x.shape
        filter_weights = self.sigmoid(self.filter_weights) * self.triangular_filter_bank
        f = filter_weights
        x = x.permute(1,0,3,2).contiguous().view(n_channels,batch_size * length,input_filter_dim)


        x = torch.bmm(x,f)
        x = x.transpose(1,2)
        x = self.bn(x)
        x = x.transpose(1,2)

        x = x.view(n_channels,batch_size,length,self.filter_dim)
        x = x.permute(1, 0, 3, 2)
        return x

class SpectrogramEncoder(nn.Module):

    def __init__(self, input_dim=129, filter_dim=20, n_channels=1, hidden_layers=64, bidir=True,
                 dropout=0.25, dropout2d=0.):
        """
        input_dim (int): size of the original filter bank
        filter_dim (int): size of the fitted filter bank
        n_channels (int): number of distinct signals (i.e. number of channels in the time-frequency representation)
        context_size (int): context size of the attention module
        hidden_layers (int): Hidden layers in the GRU module
        bidir  (bool): Use bidirectionnal gru ?
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2D = nn.Dropout2d(dropout2d)
        self.hidden_size = hidden_layers
        self.num_layers = 1
        self.filter_bank = FilterBankLayer(input_dim, filter_dim, n_channels)
        self.gru = GRU(bidirectional=bidir,input_size=filter_dim * n_channels,
                        num_layers=self.num_layers,batch_first=True,hidden_size=self.hidden_size)

    def forward(self, x):
        """
        x (tensor : batch_size,n_channels,filter_bank_size,time) :
                n_channels : number of signal (for instance EEG,EOG, EMG channels)
                filter_bank_size : frequency resolution of the initial time-frequency representation
                time: number of window/timesteps in the time-frequency representation
        returns output (tensor: batch_size, context_size*(1+bidir)), attention weights (tensor: batch_size, Time)
        """
        batch_size, n_channels, filter_size, spectrogram_length = x.shape
        x = self.filter_bank(x)
        x = x.contiguous().view(batch_size, -1, spectrogram_length).transpose(1, 2)
        x = self.dropout(x)
        x , _ = self.gru(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return x



class SeqSleepNetEpochEncoder(EpochEncoder):
    """ Class to follow in order to use below train function

    Essentially: init and forward ok
    - get_args: to retrieve args for forward function from a batch of data (out of dataloader)
    - get_transform: transformation to apply to the dataset
    - get_size_input: in order to adjust size of layers (from a batch of data size)
    """

    def __init__(self, group, net_parameters):
        """

        groups: (dict) Description of the groups (similar to Dataset.group_description)
        normalization_parameters: (DatasetNormalizationPipeline) ust be compatible with group descriptions
        net_parameters: (dict) parameters to be feed to modify the architecture of the net
        """
        super(SeqSleepNetEpochEncoder, self).__init__(group, net_parameters)

    def init_encoder(self):
        n_channels, n_freq_filters, seq_length = tuple(self.group['encoder_input_shape'][2:])
        self.encoder = SpectrogramEncoder(input_dim=n_freq_filters,
                                          n_channels=n_channels,
                                          **self.net_parameters)
        self.output_size = self.net_parameters['hidden_layers'] * (
                1 + self.net_parameters['bidir'])

    def forward(self, x):
        """
        Reshape and forward the EEG epochs
        x: (batch_size*temporal_context,n_channels,temporal_context)
        returns (batch_size*temporal_context,feature_map_size)
        """

        batch_size, temporal_context, n_channels, n_filters, spectrogram_length = x.size()
        x = x.contiguous().view(
            (batch_size * temporal_context, n_channels, n_filters, spectrogram_length))
        x = self.encoder.forward(x)
        return x