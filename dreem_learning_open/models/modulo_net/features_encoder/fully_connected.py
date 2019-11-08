from collections import OrderedDict

import torch
from torch import nn


class FullyConnected(nn.Module):

    def __init__(self, features, layers=None, dropout=0.):
        super(FullyConnected, self).__init__()
        print('Layers:', layers)
        input_channels = 0
        for feature in features:
            input_channels += features[feature]['shape'][0]
        self.dropout = nn.Dropout(dropout)

        if isinstance(layers, list):
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        OrderedDict([
                            ("linear_{}".format(i), nn.Linear(
                                in_features=input_channels if i == 0 else layers[i - 1],
                                out_features=n_dim
                            )),
                            ("relu_{}".format(i), nn.ReLU())
                        ])
                    ) for i, n_dim in enumerate(layers)
                ]
            )
            self.out_features = layers[-1]
        elif layers is None:
            self.layers = [nn.Identity()]
            self.out_features = input_channels

    def forward(self, x):
        features = []
        for feature in x:
            features += [x[feature]]
        features = torch.cat(features, -1)

        for layer in self.layers:
            features = self.dropout(features)
            features = layer(features)

        return features
