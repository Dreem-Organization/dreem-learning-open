import hashlib
import json
import os
import shutil

import h5py
import numpy as np
import tqdm
from joblib import Parallel, delayed

from ..preprocessings.epoch_features_processing import epoch_features
from ..preprocessings.signal_processing import signal_processings
from ..utils.utils import get_group_description_from_record_description


def compute_memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


def processs_group(h5, group_description, truncate_to_second=None):
    record_description = json.loads(h5.attrs['description'])
    record_description = {elt['path']: elt for elt in record_description}
    signals = []
    signals_properties = []
    for signal in group_description['signals']:
        if isinstance(signal, dict):
            processed_signals, properties = processs_group(h5, signal)
            signals_properties += [properties]
            signals += [processed_signals]

        else:  # base signal
            signals += [np.expand_dims(h5[signal][:], -1)]
            signals_properties += [
                {'fs': record_description[signal]['fs'], 'padding': 0}
            ]

            if truncate_to_second:
                sig_duration = (len(signals[-1]) // signals_properties[-1]['fs']) * \
                    signals_properties[-1]['fs']
                sig_duration = int(sig_duration)
                signals[-1] = signals[-1][:sig_duration]

    fs = [sig['fs'] for sig in signals_properties]
    padding = [sig['padding'] for sig in signals_properties]

    assert len(set(fs)) == 1, 'frequencies are not the same within the group'
    assert len(set(padding)) == 1, 'paddings are not the same within the group'
    signals_properties = {'fs': fs[0], 'padding': padding[0]}
    signals = np.concatenate(signals, -1)
    for operation in group_description['processings']:
        signals, signals_properties = signal_processings[operation['type']](
            signals, signals_properties, **operation['args'])

    signals_properties['shape'] = signals.shape
    return signals, signals_properties


def compute_epoch_features(h5, features_description, signals):
    if len(signals) == 0:
        raise ValueError('Signals have to be provided to compute the epoch features')
    record_description = json.loads(h5.attrs['description'])
    record_description = {elt['path']: elt for elt in record_description}

    epoch_processing_function = epoch_features[features_description['type']]
    signals_tp = []
    for signal in signals:
        signals_tp += [np.expand_dims(h5[signal][:], -1)]
        signals_properties = {'fs': record_description[signal]['fs'], 'padding': 0,
                              'filename': os.path.basename(h5.filename).replace('.h5', '')}
    signals_tp = np.concatenate(signals_tp, -1)
    if 'signals_preprocessing' in features_description:
        for operation in features_description['signals_preprocessing']:
            signals_tp, signals_properties = signal_processings[operation['type']](
                signals, signals_properties, **operation['args'])

    return epoch_processing_function(signals_tp, signals_properties, **features_description['args'])


def h5_to_memmaps(records, memmap_directory, memmap_description, parallel=True, force=False,
                  truncate_to_second=False, error_tolerant=False, remove_hypnogram=False):
    pipeline_hash = hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]

    assert len(
        records) > 0, "No records found, check your data path in dreem_learning_open.settings"

    def process_record(record):
        print(record)
        try:
            if isinstance(record, str):
                record_name = os.path.basename(record).replace('.h5', '')
            elif isinstance(record, h5py.File):
                record_name = os.path.basename(record.filename).replace('.h5', '')

            save_directory = os.path.join(memmap_directory, pipeline_hash, record_name)
            if not os.path.exists(save_directory):  # create it
                os.makedirs(os.path.join(save_directory, 'signals'))
                os.makedirs(os.path.join(save_directory, 'features'))
            else:
                if force is True:  # recompute anyway
                    shutil.rmtree(save_directory)
                elif len(os.listdir(save_directory)) != 5:  # corrupted or not finished
                    shutil.rmtree(save_directory)
                    os.makedirs(os.path.join(save_directory, 'signals'))
                    os.makedirs(os.path.join(save_directory, 'features'))
                else:  # do not recompute
                    return

            if not os.path.isfile(os.path.join(save_directory, 'properties.json')):
                with h5py.File(record, 'r') if isinstance(record, str) else record as h5:
                    signals_duration = None
                    # processing signals
                    groups = {'properties': {}, 'signals': {}}
                    # process each group
                    for group_description in memmap_description['signals']:
                        group_name = group_description['name']
                        groups[group_name] = {}
                        groups['signals'][group_name], groups['properties'][
                            group_name] = processs_group(
                            h5, group_description, truncate_to_second=truncate_to_second)

                    durations = []
                    # check shapes
                    for group_name, group in groups['properties'].items():
                        durations += [group['shape'][0] // group['fs']]
                    # check all the groups have the same duration

                    if len(set(durations)) > 1:

                        min_duration = int(np.min(durations))
                        durations = []
                        for group_name, group in groups['properties'].items():
                            group_fs = group['fs']
                            groups['signals'][group_name] = groups['signals'][group_name][
                                :(min_duration * group_fs)]
                            group['shape'] = groups['signals'][group_name].shape
                            durations += [group['shape'][0] // group['fs']]

                    assert len(set(durations)) == 1 or len(groups['signals']) == 0, print(
                        "Invalid signals duration or umber of groups")

                    if len(durations) > 0:
                        signals_duration = durations[0]
                        durations = []

                    # computing epoch level features_data
                    computed_features = []
                    for feature in memmap_description['features']:
                        computed_features += [{
                            'name': feature['name'],
                            'value': compute_epoch_features(h5, feature['processing'],
                                                            feature['signals'])
                        }]
                    for feature in computed_features:
                        if signals_duration is not None:
                            assert feature['value'].shape[
                                0] == signals_duration // 30, 'Invalid features_data shape for ' + \
                                feature[
                                'name']
                            durations += [feature['value'].shape[0] * 30]
                        else:
                            durations += [feature['value'].shape[0] * 30]

                    assert len(set(durations)) == 1 or len(computed_features) == 0, print(
                        record_name)
                    if len(durations) > 0:
                        signals_duration = durations[0]

                    # Hypnogram
                    if 'hypnogram' in h5 and remove_hypnogram is False:
                        assert signals_duration % 30 == 0, print(record_name)
                        hypnogram = h5['hypnogram'][:]
                        hypnogra_provided = True
                    else:
                        truncated_duration = int(signals_duration // 30)
                        hypnogram = np.array([-1] * truncated_duration)
                        hypnogra_provided = False

                    # add padding to hypnogram
                    hypnogram_duration = int(hypnogram.shape[0] * 30)
                    if hypnogram_duration != signals_duration:
                        hypnogram_padding = int((signals_duration - hypnogram_duration) // 2 / 30)
                        hypnogram = np.concatenate(
                            [np.array([-1] * hypnogram_padding), hypnogram,
                             np.array([-1] * hypnogram_padding)])

                    features_description = {}
                    for feature in computed_features:
                        features_description[feature['name']] = {"shape": feature['value'].shape}
                        feature_arr = np.memmap(
                            os.path.join(save_directory, 'features', feature['name'] + '.mm'),
                            dtype='float32',
                            mode='w+',
                            shape=feature['value'].shape)
                        feature_arr[:] = feature['value']

                    for group_name, group in groups['signals'].items():
                        if hypnogra_provided:
                            group_arr = np.memmap(os.path.join(save_directory, 'signals', group_name + '.mm'),
                                                  dtype='float32',
                                                  mode='w+',
                                                  shape=group.shape)
                            group_arr[:] = group
                        else:

                            shape = groups['properties'][group_name]
                            shape['shape'] = (
                                truncated_duration * groups['properties'][group_name]['fs'] * 30,
                                *shape['shape'][1:])
                            group_arr = np.memmap(os.path.join(save_directory, 'signals', group_name + '.mm'),
                                                  dtype='float32',
                                                  mode='w+',
                                                  shape=shape['shape'])
                            group_arr[:] = group[:shape['shape'][0]]

                    hypnogram_memmap = np.memmap(os.path.join(save_directory, 'hypno' + '.mm'), dtype='float32',
                                                 mode='w+',
                                                 shape=hypnogram.shape)
                    hypnogram_memmap[:] = hypnogram

                    with open(os.path.join(save_directory, 'features_description.json'), 'w') as f:
                        json.dump(features_description, f, indent=2)

                    with open(os.path.join(save_directory, 'properties.json'), 'w') as f:
                        json.dump(groups['properties'], f, indent=2)
        except OSError:
            shutil.rmtree(save_directory)
            print(record_name)

        except Exception as e:
            if error_tolerant:
                shutil.rmtree(save_directory)
                print(record_name)
                print(e)
            else:
                raise e

    if not os.path.exists(os.path.join(memmap_directory, pipeline_hash)):
        os.makedirs(os.path.join(memmap_directory, pipeline_hash))

    if parallel is True:
        print('###########################')
        print('Running the jobs in parralel')
        Parallel(n_jobs=-1, verbose=10)(delayed(process_record)(record) for record in records)
    else:
        for record in tqdm.tqdm(records):
            process_record(record)

    groups_description = []
    for record_name in os.listdir(os.path.join(memmap_directory, pipeline_hash)):
        save_directory = os.path.join(memmap_directory, pipeline_hash, record_name)
        if os.path.isdir(save_directory):
            with open(os.path.join(save_directory, 'properties.json')) as f:
                record_description = json.load(f)
                groups_description += [
                    get_group_description_from_record_description(record_description)]

            with open(os.path.join(save_directory, 'features_description.json')) as f:
                features_description = json.load(f)
                for feature in features_description:
                    features_description[feature]['shape'] = features_description[feature]['shape'][
                        1:]

    groups_description_hash = [hash(json.dumps(d)) for d in groups_description]
    assert len(np.unique(groups_description_hash)) == 1
    with open(os.path.join(memmap_directory, pipeline_hash, 'groups_description.json'), 'w') as f:
        json.dump(groups_description[0], f, indent=2)

    with open(os.path.join(memmap_directory, pipeline_hash, 'features_description.json'), 'w') as f:
        json.dump(features_description, f, indent=2)

    with open(os.path.join(memmap_directory, pipeline_hash, 'memmap_description.json'), 'w') as f:
        json.dump(memmap_description, f, indent=2)

    return os.path.join(memmap_directory, pipeline_hash), groups_description[0], features_description
