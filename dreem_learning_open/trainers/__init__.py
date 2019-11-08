from torch import nn
from torch.optim import Adam, SGD

optimizers = {
    "adam": Adam,
    'sgd': SGD
}

loss_functions = {
    'cross_entropy': nn.CrossEntropyLoss,
}

from .trainer import Trainer

__all__ = [
    "Trainer"]
