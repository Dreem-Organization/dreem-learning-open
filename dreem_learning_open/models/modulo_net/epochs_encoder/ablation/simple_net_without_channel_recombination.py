from torch import nn

from ..epoch_encoder import EpochEncoder


class SimpleSpectrogramEncoder(nn.Module):

    def __init__(self, input_dim=129, filter_dim=30, n_channels=1, hidden_layers=64, bidir=True,
                 num_layers=1,
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
        self.filter_reduction = nn.Linear(input_dim, filter_dim)

        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=filter_dim * n_channels, hidden_size=self.hidden_size,
                          bidirectional=bidir, num_layers=self.num_layers, batch_first=True)

    def forward(self, x):
        """
        x (tensor : batch_size,n_channels,filter_bank_size,time) :
                n_channels : number of signal (for instance EEG,EOG, EMG channels)
                filter_bank_size : frequency resolution of the initial time-frequency representation
                time: number of window/timesteps in the time-frequency representation
        returns output (tensor: batch_size, context_size*(1+bidir)), attention weights (tensor: batch_size, Time)
        """
        batch_size, n_channels, filter_size, spectrogram_length = x.shape

        x_tp = self.dropout2D(x)
        # Separated convolution on filter + channels (spatio-frequential map)
        x = x.transpose(2, 3).contiguous()
        x = self.filter_reduction(x)
        x = x.transpose(2, 3)
        # x = self.dropout(x)
        # Dropout2D
        x = self.dropout(x)

        # Reshape for GRU
        x = x.contiguous().view(batch_size, -1, spectrogram_length).transpose(1, 2)

        x, h = self.gru(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return x, h


class SimpleSleepNetEpochEncoderWithoutChannelRecombination(EpochEncoder):
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
        super(SimpleSleepNetEpochEncoderWithoutChannelRecombination, self).__init__(group,
                                                                                    net_parameters)

    def init_encoder(self):
        n_channels, n_freq_filters, seq_length = tuple(self.group['encoder_input_shape'][2:])
        self.encoder = SimpleSpectrogramEncoder(input_dim=n_freq_filters,
                                                n_channels=n_channels,
                                                **self.net_parameters)

    def forward(self, x):
        """
        Reshape and forward the EEG epochs
        x: (batch_size*temporal_context,n_channels,temporal_context)
        returns (batch_size*temporal_context,feature_map_size)
        """

        batch_size, temporal_context, n_channels, n_filters, spectrogram_length = x.size()

        x = x.view(batch_size, temporal_context, n_channels, n_filters, spectrogram_length)
        x = x.contiguous().view(
            (batch_size * temporal_context, n_channels, n_filters, spectrogram_length))
        x, hidden_states = self.encoder.forward(x)
        return x
