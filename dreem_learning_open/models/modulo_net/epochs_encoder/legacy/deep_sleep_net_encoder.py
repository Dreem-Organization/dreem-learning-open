import torch
from torch import nn

from dreem_learning_open.models.modulo_net.epochs_encoder.epoch_encoder import EpochEncoder


class Encoder(nn.Module):

    def __init__(self, n_channels, fs, dropout=0.5, cells=64, dropout2d=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        momentum = 0.1
        track_running_stats = True
        # First branch
        self.dropout_2d_rate = dropout2d
        self.dropout2d = nn.Dropout2d(dropout2d)
        self.conv_1_1 = nn.Conv1d(n_channels, cells, kernel_size=int(fs / 2),
                                  stride=int(fs / 16), padding=int(fs / 4))
        self.bn_1_1 = nn.BatchNorm1d(cells, momentum=momentum,
                                     track_running_stats=track_running_stats)
        self.pool_1_1 = nn.MaxPool1d(kernel_size=8, stride=7)

        self.conv_1_2 = nn.Conv1d(cells, cells * 2, kernel_size=8, padding=4)
        self.bn_1_2 = nn.BatchNorm1d(cells * 2, momentum=momentum,
                                     track_running_stats=track_running_stats)

        self.conv_1_3 = nn.Conv1d(cells * 2, cells * 2, kernel_size=8, padding=4)
        self.bn_1_3 = nn.BatchNorm1d(cells * 2, momentum=momentum,
                                     track_running_stats=track_running_stats)

        self.conv_1_4 = nn.Conv1d(cells * 2, cells * 2, kernel_size=8, padding=4)
        self.bn_1_4 = nn.BatchNorm1d(cells * 2, momentum=momentum,
                                     track_running_stats=track_running_stats)

        self.pool_1_2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        # Second branch
        self.conv_2_1 = nn.Conv1d(n_channels, cells, kernel_size=int(fs * 2),
                                  stride=int(fs / 2), padding=fs)
        self.bn_2_1 = nn.BatchNorm1d(cells, momentum=momentum,
                                     track_running_stats=track_running_stats)
        self.pool_2_1 = nn.MaxPool1d(kernel_size=4, stride=3)

        self.conv_2_2 = nn.Conv1d(cells, cells * 2, kernel_size=6, padding=2)
        self.bn_2_2 = nn.BatchNorm1d(cells * 2, momentum=momentum,
                                     track_running_stats=track_running_stats)

        self.conv_2_3 = nn.Conv1d(cells * 2, cells * 2, kernel_size=6, padding=2)
        self.bn_2_3 = nn.BatchNorm1d(cells * 2, momentum=momentum,
                                     track_running_stats=track_running_stats)

        self.conv_2_4 = nn.Conv1d(cells * 2, cells * 2, kernel_size=6, padding=2)
        self.bn_2_4 = nn.BatchNorm1d(cells * 2, momentum=momentum,
                                     track_running_stats=track_running_stats)

        self.pool_2_2 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # First branch
        if self.dropout_2d_rate > 0:
            x = self.dropout2d(x)
        x_1 = self.conv_1_1(x)
        x_1 = self.bn_1_1(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.pool_1_1(x_1)
        x_1 = self.dropout(x_1)

        x_1 = self.conv_1_2(x_1)
        x_1 = self.bn_1_2(x_1)
        x_1 = self.relu(x_1)

        x_1 = self.conv_1_3(x_1)
        x_1 = self.bn_1_3(x_1)
        x_1 = self.relu(x_1)

        x_1 = self.conv_1_4(x_1)
        x_1 = self.bn_1_4(x_1)
        x_1 = self.relu(x_1)

        x_1 = self.pool_1_2(x_1)

        # second branch
        x_2 = self.conv_2_1(x)
        x_2 = self.bn_2_1(x_2)
        x_2 = self.relu(x_2)
        x_2 = self.pool_2_1(x_2)
        x_2 = self.dropout(x_2)

        x_2 = self.conv_2_2(x_2)
        x_2 = self.bn_2_2(x_2)
        x_2 = self.relu(x_2)

        x_2 = self.conv_2_3(x_2)
        x_2 = self.bn_2_3(x_2)
        x_2 = self.relu(x_2)

        x_2 = self.conv_2_4(x_2)
        x_2 = self.bn_2_4(x_2)
        x_2 = self.relu(x_2)
        x_2 = self.pool_2_2(x_2)

        x_1 = x_1[:, :, :-1]

        x_cat = torch.cat([x_1, x_2], 1)
        x_cat = self.dropout(x_cat)
        return x_cat


class DeepSleepNetEpochEncoder(EpochEncoder):
    """ Class to follow in order to use below train function

    Essentially: init and forward ok
    - get_args: to retrieve args for forward function from a batch of data (out of dataloader)
    - get_transform: transformation to apply to the dataset
    - get_size_input: in order to adjust size of layers (from a batch of data size)
    """

    @staticmethod
    def defaut_net_parameters():
        return {'dropout': 0.5, 'cells': 32}

    def __init__(self, group, net_parameters):
        """

        groups: (dict) Description of the groups (similar to Dataset.group_description)
        normalization_parameters: (DatasetNormalizationPipeline) ust be compatible with group descriptions
        net_parameters: (dict) parameters to be feed to modify the architecture of the net
        """
        super(DeepSleepNetEpochEncoder, self).__init__(group, net_parameters)

    def init_encoder(self):
        n_channels = self.group['encoder_input_shape'][2]
        self.encoder = Encoder(n_channels=n_channels, fs=self.group['fs'],
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
