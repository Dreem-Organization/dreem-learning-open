import json
import os

import h5py
import numpy as np
import pyedflib

from dreem_learning_open.utils.utils import standardize_signals_durations


def get_sleep_stages(annotation_file):
    """
    Extract the sleep stages from an annotation file
    annotation_file : (str) path to EDF annotation file
    returns stages: list of sleep stages
            time_begin: beginning of hypno
            time_end: end of hypno
    """
    with pyedflib.EdfReader(annotation_file) as annotation_file:
        annotations = annotation_file.readAnnotations()
        stage_idx = []
        stages = []
        for i, annot in enumerate(annotations[2]):
            for stage in stages_lookup:
                if stage in annot:
                    stage_idx += [i]
                    stages += [stages_lookup[stage]]
        time_begin, time_end = annotations[0][stage_idx[0]], annotations[0][stage_idx[-1]] + annotations[1][
            stage_idx[-1]]
    return stages, time_begin, time_end


def get_annotation(annotation_file, annotation_name, sampling_freq=64):
    """
    Extract annotation from an EDF file
    annotation_file : EDF handle
    annotation_name :(str) name of the annoation to get
    sampling_freq : (int) sampling freq to use to build the event binary representation
    """

    annotations = annotation_file.readAnnotations()
    result = np.zeros(annotation_file.file_duration * sampling_freq)
    annot_idx = np.where(annotations[2] == annotation_name)[0]
    time_begins, durations = [], []
    for idx in annot_idx:
        time_begins += [annotations[0][idx]]
        durations += [annotations[1][idx]]
        time_begin = int(annotations[0][idx] * sampling_freq)
        time_end = time_begin + int(annotations[1][idx] * sampling_freq)
        result[time_begin:time_end] = 1
    return result, time_begins, durations


def std_name(name):
    """
    standardize channels name
    name:
    """
    remove = ['EEG ', '-CLE', '-LER', 'EOG ', 'EMG ', 'ECG ', 'Resp ']
    for word in remove:
        name = name.replace(word, '')
    name = name.lower()
    return name


def to_h5(record_file, annotation_files, h5_target_directory, signals, crop_record=True):
    """
    Format a MASS EDF record and its annotation to a standardized h5
    record_file :(str)
    annotation_files :(list of str) the hypnogram has to be in the first annotation file
    h5_target :(str)
    crop_record : (bool)
    """
    description = []
    events_description = []
    with pyedflib.EdfReader(record_file) as data:
        with h5py.File(h5_target_directory, "w", driver="core") as h5_target:
            signal_labels = {key: value for value, key in enumerate(data.getSignalLabels())}

            hypno, time_begin, time_end = get_sleep_stages(annotation_files[0])
            h5_target['hypnogram'] = np.array(hypno).astype(int)

            # Add signal
            h5_target.create_group('signals')
            for group_name, signals_list in signals.items():
                group_name = group_name.lower()
                h5_target['signals'].create_group(group_name)
                mod_fs = None
                mod_unit = None
                for signal in signals_list:
                    signal_idx = signal_labels[signal]
                    signal = std_name(signal)
                    if mod_fs is None:
                        mod_fs = int(data.getSignalHeader(signal_idx)['sample_rate'])
                        mod_unit = data.getSignalHeader(signal_idx)['dimension']
                    if mod_fs is not None:
                        signal_path = "signals/" + group_name + '/' + signal
                        if mod_fs == data.getSignalHeader(signal_idx)['sample_rate'] and mod_unit == \
                                data.getSignalHeader(signal_idx)['dimension']:
                            if crop_record:
                                begin_idx = int(time_begin * mod_fs)
                                end_idx = int(time_end * mod_fs)
                                x = data.readSignal(signal_idx)[begin_idx:end_idx].astype(np.float32)
                                h5_target.create_dataset(
                                    signal_path,
                                    data=x,
                                    compression='gzip')
                            else:
                                x = data.readSignal(signal_idx).astype(np.float32)
                                h5_target.create_dataset(signal_path, data=x,
                                                         compression='gzip')
                            signal_description = {"fs": mod_fs, "unit": mod_unit,
                                                  "path": signal_path, 'name': signal,
                                                  'domain': group_name, "default": True}
                            description += [signal_description]
                        else:
                            print('Signal: ', signal, 'has invalid frequency or dimension for the modality')

                h5_target["signals/" + group_name].attrs['fs'] = mod_fs
                h5_target["signals/" + group_name].attrs['unit'] = mod_unit

            # add events
            h5_target.create_group('events')
            for annotation_file in annotation_files:
                with pyedflib.EdfReader(annotation_file) as annotations:
                    for name, properties in SS3_events.items():
                        result, time_begins, durations = get_annotation(annotations, name, properties['fs'])
                        result = np.array(result).astype(int)
                        event_name = properties['name']
                        event_path = 'events/' + event_name + '/'
                        if crop_record:
                            begin_idx = int(time_begin * properties['fs'])
                            end_idx = int(time_end * properties['fs'])
                            h5_target.create_dataset(event_path + 'binary', data=result[begin_idx:end_idx],
                                                     compression='gzip', dtype='i8')
                        else:
                            h5_target.create_dataset(event_path + 'binary', data=result,
                                                     compression='gzip', dtype='i8')
                        h5_target.create_dataset(event_path + 'begin', data=time_begins,
                                                 compression='gzip')
                        h5_target.create_dataset(event_path + 'duration', data=durations,
                                                 compression='gzip')
                        event_description = {'name': event_name, 'fs': properties['fs'], 'path': event_path}
                        events_description += [event_description]

            h5_target.attrs.create('description', json.dumps(description), dtype=np.dtype('S32768'))
            h5_target.attrs.create('events_description', json.dumps(events_description), dtype=np.dtype('S32768'))

            # truncate file
            h5_target.attrs['duration'] = standardize_signals_durations(h5_target)

            h5_target.close()
            print('Sucess: ', h5_target_directory)
            return True


# List of MASS event
SS3_events = {'<Event channel="EEG C4-LER" groupName="MicroArousal" name="CARSM expert" scoringType="expert"/>': {
    'name': 'micro_arousal', 'fs': 16},
    '<Event channel="EMG Ant Tibial L" groupName="PLMS" name="CARSM detector" scoringType="automatic"/>': {
        'name': 'PLMS Left', 'fs': 16},
    '<Event channel="EMG Ant Tibial R" groupName="PLMS" name="CARSM detector" scoringType="automatic"/>': {
        'name': 'PLMS Right', 'fs': 16},
    '<Event channel="Resp Cannula" groupName="Hypopnea" name="CARSM expert" scoringType="expert"/>': {
        'name': 'Hypopnea', 'fs': 16},
    '<Event channel="Resp Cannula" groupName="ObstructiveApnea" name="CARSM expert" scoringType="expert"/>':
        {'name': 'ObstructiveApnea', 'fs': 16}
}

stages_lookup = {'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage ?': -1,
                 'Sleep stage R': 4, 'Sleep stage W': 0}

if __name__ == "__main__":

    from dreem_learning_open.settings import MASS_SETTINGS
    from joblib import Parallel, delayed

    records_directory, annotations_directory, h5_directory = MASS_SETTINGS['edf_directory'], \
                                                             MASS_SETTINGS[
        'annotations_directory'], MASS_SETTINGS['h5_directory']

    if not os.path.exists(h5_directory):
        os.mkdir(h5_directory)

    records = os.listdir(records_directory)
    annotations = os.listdir(annotations_directory)
    records_name = [x.split(" ")[0] for x in records]
    annotations_name = [x.split(" ")[0] for x in annotations]
    records_with_annotations = set(records_name).intersection(set(annotations_name))
    records = [x for x in records if any([y in x for y in records_with_annotations])]
    annotations = [x for x in annotations if any([y in x for y in records_with_annotations])]
    records.sort()
    annotations.sort()
    assert len(annotations) == len(records)

    parallel = False


    def record_to_h5_full(record, annotation):
        with pyedflib.EdfReader(records_directory + record) as f:
            channels = f.getSignalLabels()
        signals_prefix = ['EEG', 'EMG', 'ECG', 'EOG', 'Belt', 'Resp', 'SpO2']
        signals = {}
        for channel in channels:
            for prefix in signals_prefix:
                if prefix in channel:
                    if prefix not in signals:
                        signals[prefix] = []
                    signals[prefix] += [channel]
                    break

        output_file = h5_directory + record
        output_file = output_file.replace('.edf', '.h5')
        to_h5(records_directory + record, [annotations_directory + annotation], output_file,
              signals=signals)


    if parallel is True:
        Parallel(n_jobs=-1)(
            delayed(record_to_h5_full)(records[i], annotations[i]) for i in
            range(len(records)))
    else:
        for i in range(len(records)):
            record_to_h5_full(records[i], annotations[i])
