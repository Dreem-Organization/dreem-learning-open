import json
import os

import h5py
import mne
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
    annotation = mne.read_annotations(annotation_file)

    duration = annotation.onset
    labels = annotation.description
    duration = np.diff(duration)
    labels = labels[:-1]
    stages = []
    for i, (dur, annot) in enumerate(zip(duration, labels)):
        for stage in stages_lookup:
            if stage in annot:
                stages += [stages_lookup[stage]] * int(dur // 30)
    stages_as_array = np.array(stages)
    first_sleep, last_sleep = np.where(stages_as_array > 0)[0][0], np.where(stages_as_array > 0)[0][
        -1]
    first_sleep, last_sleep = max(0, first_sleep - 60), min(len(stages_as_array), last_sleep + 60)
    stages = stages[first_sleep:last_sleep]

    return stages, first_sleep * 30, last_sleep * 30


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


def to_h5(record_file, annotation_files, h5_target_directory, signals, crop_record=True,
          force = False):
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
        if force or not os.path.exists(h5_target_directory):
            with h5py.File(h5_target_directory, "w", driver="core") as h5_target:
                signal_labels = {key: value for value, key in enumerate(data.getSignalLabels())}

                hypno, time_begin, time_end = get_sleep_stages(annotation_files)
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

                h5_target.attrs.create('description', json.dumps(description), dtype=np.dtype('S32768'))
                h5_target.attrs.create('events_description', json.dumps(events_description), dtype=np.dtype('S32768'))

                # truncate file
                h5_target.attrs['duration'] = standardize_signals_durations(h5_target)

                h5_target.close()
                print('Sucess: ', h5_target_directory)
                return True
    return True


# List of MASS event

stages_lookup = {'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3,
                    'Movement time': -1,
                 'Sleep stage ?': -1,
                 'Sleep stage R': 4, 'Sleep stage W': 0}

if __name__ == "__main__":

    from dreem_learning_open.settings import SLEEP_EDF_SETTINGS
    from joblib import Parallel, delayed

    records_directory, h5_directory = SLEEP_EDF_SETTINGS['edf_directory'], SLEEP_EDF_SETTINGS[
        'h5_directory']
    records = {}
    for directory in records_directory:
        records.update({x:directory for x in os.listdir(directory) if '-PSG.edf' in x})

    record_annotation = {}
    for record_name, directory in records.items():
        record_id =  record_name[:7]
        annotations_in_directory = [x for x in os.listdir(directory) if '-Hypnogram.edf' in x
        and record_id in x]
        assert len(annotations_in_directory) == 1
        record_annotation[f'{directory}/{record_name}'] = f'{directory}/' \
                                                          f'{annotations_in_directory[0]}'


    if not os.path.exists(h5_directory):
        os.mkdir(h5_directory)



    parallel = False


    def record_to_h5_full(record, annotation,force = False):
        with pyedflib.EdfReader(record) as f:
            channels = f.getSignalLabels()
        signals_prefix = ['EEG', 'EMG', 'ECG', 'EOG']
        signals = {}
        for channel in channels:
            for prefix in signals_prefix:
                if prefix in channel:
                    if prefix not in signals:
                        signals[prefix] = []
                    signals[prefix] += [channel]
                    break

        output_file = h5_directory + record.split('/')[-1].replace('.edf', '.h5')

        to_h5(record,  annotation, output_file,
              signals=signals,force=force)


    if parallel is True:
        Parallel(n_jobs=-1)(
            delayed(record_to_h5_full)(record, annotation) for record,annotation in
            record_annotation.items())
    else:
        for record, annotation in record_annotation.items():
            print(record,annotation)
            record_to_h5_full(record, annotation,force=False)
