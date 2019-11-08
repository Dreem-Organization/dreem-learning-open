import json
import os
import shutil

import numpy as np

from ..preprocessings.h5_to_memmap import h5_to_memmaps
from .descriptions import three_groups_record_description
from .utils import generate_fake_record
import pytest
from ..preprocessings.epoch_features_processing import cyclic_index_window, index_window

features_memmap = {}
features_memmap['signals'] = []
features_memmap['features'] = [
    {
        'name': 'epoch_index',
        'processing': {'type': 'index_window', 'args': {
            'increment_duration': 30,
            'padding_duration': 900
        }},
        'signals': ["signals/eeg/eeg1"],
        "signals_name": ["signals/eeg/eeg1"]
    },
    {
        'name': 'epoch_cycle_index',
        'processing': {'type': 'cycle_index_window', 'args': {
            'increment_duration': 30,
            'padding_duration': 900
        }},
        'signals': ["signals/eeg/eeg1"],
        "signals_name": ["signals/eeg/eeg1"]
    }
]

features_memmap_wrong_signal_name = {}
features_memmap_wrong_signal_name['signals'] = []
features_memmap_wrong_signal_name['features'] = [
    {
        'name': 'epoch_index',
        'processing': {'type': 'index_window', 'args': {
            'increment_duration': 30,
            'padding_duration': 900
        }},
        'signals': ["signals/eeg/eeg_fake"],
        'signals_name': ["signals/eeg/eeg_fake"]
    },
    {
        'name': 'epoch_cycle_index',
        'processing': {'type': 'cycle_index_window', 'args': {
            'increment_duration': 30,
            'padding_duration': 900
        }},
        'signals': ["signals/eeg/eeg1"],
        'signals_name': ["signals/eeg/eeg_fake"]
    }
]


def test_memmap_with_features_only():
    """
    Check that a memmap with only features caan be created
    """
    generate_fake_record(three_groups_record_description)
    memmaps_description = features_memmap
    record_directory = \
    h5_to_memmaps(['/tmp/fake.h5'], '/tmp/memmap_test/', memmap_description=memmaps_description,
                  parallel=False)[0]

    record_directory = record_directory + \
                       [record for record in os.listdir(record_directory) if '.' not in record][
                           0] + '/'
    with open(record_directory + 'features_description.json') as f:
        features_description = json.load(f)

    shape = tuple(features_description['epoch_cycle_index']['shape'])
    computed_epoch_cycle_index = np.memmap(record_directory + 'features/epoch_cycle_index.mm',
                                           mode='r',
                                           dtype='float32',
                                           shape=shape)

    assert len(computed_epoch_cycle_index.shape) == 2

    fake_sig = np.arange(0, computed_epoch_cycle_index.shape[0] - 60)
    cyclic = cyclic_index_window(fake_sig, {'fs': 1}, increment_duration=1,
                                 padding_duration=30).astype(np.float32)
    assert np.linalg.norm(computed_epoch_cycle_index - cyclic) == 0

    shape = tuple(features_description['epoch_index']['shape'])
    epoch_index = np.memmap(record_directory + 'features/epoch_index.mm', mode='r',
                            dtype='float32',
                            shape=shape)
    new_epoch_index = index_window(fake_sig, {'fs': 1}, increment_duration=1,
                                   padding_duration=30).astype(np.float32)
    assert np.linalg.norm(epoch_index - new_epoch_index) == 0

    shutil.rmtree('/tmp/memmap_test/')
    os.remove('/tmp/fake.h5')


def test_memmap_with_features_and_wrong_signal_name():
    """
    Check that a memmap where features description is linked to unknown signals raise an error on creation
    """
    generate_fake_record(three_groups_record_description)
    memmaps_description = features_memmap_wrong_signal_name
    with pytest.raises(KeyError):
        h5_to_memmaps(['/tmp/fake.h5'], '/tmp/memmap_test/', memmap_description=memmaps_description,
                      parallel=False)
    shutil.rmtree('/tmp/memmap_test/')
    os.remove('/tmp/fake.h5')
