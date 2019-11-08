import numpy as np
import torch
from torch import nn


class PoolReducer(nn.Module):

    def __init__(self, groups, pool_operation='max', group_dropout=0.):
        super(PoolReducer, self).__init__()
        self.group_dropout = group_dropout
        if pool_operation == 'max':
            self.pool = lambda x: x.max(2)[0]
        elif pool_operation == 'average':
            self.pool = lambda x: x.mean(2)
        else:
            raise ValueError('pool_operation must be in ["max","average"]')

    def forward(self, x):
        features = []
        for group in x:
            tmp = self.pool(x[group])
            if np.random.uniform() < self.group_dropout:
                tmp = tmp * 0
            features += [tmp]
        features = torch.cat(features, -1)
        return features


class FlattenReducer(nn.Module):

    def __init__(self, groups, group_dropout=0.):
        self.group_dropout = group_dropout
        super(FlattenReducer, self).__init__()

    def forward(self, x):
        features = []
        for group in x:
            tmp = x[group].contiguous().view(x[group].shape[0], -1).contiguous()
            if np.random.uniform() < self.group_dropout:
                tmp = tmp * 0
            features += [tmp]
        features = torch.cat(features, -1)
        return features
