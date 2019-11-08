from torch import nn
from collections import OrderedDict


class TemporalFlattener(nn.Module):

    def __init__(self, feature_map_size, layers=None):
        super(TemporalFlattener, self).__init__()

        if isinstance(layers, list):
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        OrderedDict([
                            ("linear_{}".format(i), nn.Linear(
                                in_features=feature_map_size if i == 0 else layers[i - 1],
                                out_features=n_dim
                            )),
                            ("relu_{}".format(i), nn.ReLU())
                        ])
                    ) for i, n_dim in enumerate(layers)
                ]
            )
            self.output_size = layers[-1]
        elif layers is None:
            self.layers = [nn.Identity()]
            self.output_size = feature_map_size

    def forward(self, x, hidden_state=None):
        x = x.view(x.shape[0], 1, -1)
        for layer in self.layers:
            x = layer(x)
        return x, x
