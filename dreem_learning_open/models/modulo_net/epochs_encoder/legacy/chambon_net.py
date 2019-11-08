from collections import OrderedDict

from torch import nn

from dreem_learning_open.models.modulo_net.epochs_encoder.epoch_encoder import EpochEncoder


class Encoder(nn.Module):

    def __init__(self, n_channels, window_length, fs, dropout=0.5):
        super().__init__()
        self.flatten_shape = n_channels * (
                window_length // fs) * 8
        self.relu = nn.ReLU()
        self.encoder = nn.Sequential(
            OrderedDict([
                ("spatial_conv", nn.Linear(n_channels, n_channels)),
                ("time_conv_1",
                 nn.Conv2d(in_channels=1, kernel_size=(65, 1), out_channels=8, stride=(1, 1),
                           padding=(32, 0))),
                ('relu_1', nn.ReLU()),
                ("pool_1", nn.MaxPool2d(kernel_size=(16, 1), stride=(16, 1))),

                ("time_conv_2",
                 nn.Conv2d(in_channels=8, kernel_size=(65, 1), out_channels=8, stride=(1, 1),
                           padding=(32, 0))),
                ('relu_2', nn.ReLU()),
                ("pool_2", nn.MaxPool2d(kernel_size=(16, 1), stride=(16, 1))),
            ])
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, n_channels, window_length = x.size()
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)
        x = self.encoder.forward(x)
        x = self.dropout(x)

        return x


class ChambonEpochEncoder(EpochEncoder):
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
        super(ChambonEpochEncoder, self).__init__(group, net_parameters)

    def init_encoder(self):
        n_channels, window_length = self.group['encoder_input_shape'][2], \
                                    self.group['encoder_input_shape'][3]

        self.encoder = Encoder(n_channels=n_channels, window_length=window_length,
                               fs=self.group['fs'],
                               **self.net_parameters)

    def forward(self, x):
        """
        Reshape and forward the EEG epochs
        x: (batch_size*temporal_context,n_channels,temporal_context)
        returns (batch_size*temporal_context,feature_map_size)
        """

        batch_size, temporal_context, n_channels, window_length = x.size()
        x = x.view((-1, n_channels, window_length))
        x = self.encoder.forward(x)
        return x
