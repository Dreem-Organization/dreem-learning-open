"""
Data augmentation to use during the training
"""

import numpy as np


def kill_channel(x, p=0.5):
    if np.random.rand() < p:

        x_tp = np.zeros(shape=x.shape)
        channels = list(range(x.shape[1]))
        N_channels_to_kill = np.random.randint(1, len(channels) + 1)

        channels_to_kill = np.random.choice(channels, size=N_channels_to_kill, replace=False)
        for i in channels_to_kill:
            x_tp[:, i, :] = x[:, i, :]
        return x - x_tp
    else:
        return x


data_augmentations = {
    'kill_channel': kill_channel}


def augment_data(data, augmentation_pipeline):
    for group in augmentation_pipeline:
        group_name = group['name']
        operations = group['processing']
        for operation in operations:
            data[group_name] = data_augmentations[operation['type']](data[group_name],
                                                                     **operation['args'])
    return data
