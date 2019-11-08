import torch
from torch import nn


class EpochEncoder(nn.Module):
    """ Class to follow in order to use below train function

    Essentially: init and forward ok
    - get_args: to retrieve args for forward function from a batch of data (out of dataloader)
    """

    @staticmethod
    def defaut_net_parameters():
        return {}

    def __init__(self, group, net_parameters=None):
        """

        groups: (dict) Description of the groups (similar to Dataset.group_description)
        normalization_parameters: (DatasetNormalizationPipeline) ust be compatible with group descriptions
        net_parameters: (dict) parameters to be feed to modify the architecture of the net
        """
        super(EpochEncoder, self).__init__()
        self.group = group
        self.net_parameters = self.defaut_net_parameters()
        if net_parameters is not None:
            for param in net_parameters:
                self.net_parameters[param] = net_parameters[param]

        self.output_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_encoder()
        self.to(self.device)

    def init_encoder(self):
        print('Base encoder, no net to create')

    def forward(self, x):
        return x
