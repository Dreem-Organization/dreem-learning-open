from collections import OrderedDict

from torch import nn

from dreem_learning_open.models.modulo_net.epochs_encoder.epoch_encoder import EpochEncoder


class Encoder(nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        self.relu = nn.ReLU()
        self.encoder = nn.Sequential(
            OrderedDict([
                ("conv_1",
                 nn.Conv1d(in_channels=n_channels, kernel_size=200, out_channels=20, stride=1)),
                ('relu_1', nn.ReLU()),
                ("pool_1", nn.MaxPool1d(kernel_size=20, stride=10)),

                ("conv_2",
                 nn.Conv1d(in_channels=20, kernel_size=30, out_channels=400)),
                ('relu_2', nn.ReLU()),
                ("pool_2", nn.MaxPool1d(kernel_size=10, stride=2)),
            ])
        )

    def forward(self, x):
        x = self.encoder.forward(x)
        return x


class TsinalisEpochEncoder(EpochEncoder):
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
        super(TsinalisEpochEncoder, self).__init__(group, net_parameters)

    def init_encoder(self):
        n_channels, window_length = self.group['encoder_input_shape'][2], \
                                    self.group['encoder_input_shape'][3]

        self.encoder = Encoder(n_channels=n_channels)

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
