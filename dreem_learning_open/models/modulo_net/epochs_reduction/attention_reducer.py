import numpy as np
import torch
from torch import nn

from ..modules import Attention


class AttentionReducer(nn.Module):

    def __init__(self, groups, context_size=32, group_dropout=0., activation='tanh'):
        super(AttentionReducer, self).__init__()
        self.attentions = {}
        self.group_dropout = group_dropout
        for group in groups:
            channels = groups[group]['reducer_input_shape'][0]
            self.attentions[group] = Attention(channels, context_size, activation=activation)

        for key, module in self.attentions.items():
            self.add_module(key, module)

    def forward(self, x):
        features = []
        for group in x:
            tmp, _ = self.attentions[group](x[group].transpose(1, 2))
            if np.random.uniform() < self.group_dropout:
                tmp = tmp * 0
            features += [tmp]
        features = torch.cat(features, -1)
        return features
