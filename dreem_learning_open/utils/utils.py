import copy
import json
from functools import wraps

import numpy as np


def compute_signals_duration(record):
    durations = []
    description = json.loads(record.attrs['description'])

    for signal_description in description:
        fs = signal_description['fs']
        duration = int(record[signal_description['path']].shape[0]) // fs
        durations += [duration]
    min_duration = np.min(durations)
    return min_duration


def standardize_signals_durations(record, epoch_duration=30):
    """

    record : (h5 handle) record which signals are to be truncated
    returns signal duration (int)
    """
    durations = []
    description = json.loads(record.attrs['description'])
    events_description = json.loads(record.attrs['events_description'])

    for signal_description in description:
        fs = signal_description['fs']
        duration = int(record[signal_description['path']].shape[0]) // fs
        durations += [duration]

    for event_description in events_description:
        fs = event_description['fs']
        duration = int(record[event_description['path'] + '/binary'].shape[0]) // fs
        durations += [duration]

    durations += [len(record['hypnogram'][:]) * epoch_duration]
    min_duration = np.min(durations)
    min_duration = int(min_duration // epoch_duration) * epoch_duration
    print('Record duration:', np.round(min_duration / 60 / 60, 2), ' hours')
    print('Epoch duration:', epoch_duration, '. Number of epochs: ', record['hypnogram'].shape[0])

    for signal_description in description:
        fs = signal_description['fs']
        arr = np.array(record[signal_description['path']])[:(min_duration * fs)]
        del record[signal_description['path']]
        record[signal_description['path']] = arr

    for event_description in events_description:
        fs = event_description['fs']
        binary = np.array(record[event_description['path'] + '/binary'])[:(min_duration * fs)]
        del record[event_description['path'] + '/binary']
        record[event_description['path'] + '/binary'] = binary

    hypno = record['hypnogram'][()][:(min_duration // epoch_duration)]
    del record['hypnogram']
    record['hypnogram'] = hypno

    return min_duration


def get_group_description_from_record_description(record_description):
    groups_description = {}
    for group in record_description:
        group_description = copy.deepcopy(record_description[group])
        group_description['window_length'] = group_description['fs'] * 30
        group_description['shape'] = [group_description['window_length']] + group_description[
                                                                                'shape'][1:]
        groups_description[group] = group_description
    return groups_description


stage_correspondance = {0: "WAKE", 1: "N1", 2: "N2", 3: "DEEP", 4: "REM", -1: 'NA'}
