import json
import os
import shutil

import h5py
import numpy as np

from ..preprocessings.h5_to_memmap import h5_to_memmaps
from ..utils.utils import standardize_signals_durations


def generate_fake_hypno(transition_kernel, hypnogram_length, s_0=0):
    generated_hypno = []
    s_0 = np.array(s_0)
    for i in range(hypnogram_length):
        generated_hypno += [s_0 - 1]
        s_0 = np.random.choice(range(5), p=transition_kernel[s_0])
    return generated_hypno


def generate_fake_signal(hypno, sampling_fs, class_fs=None, random_noise=0.3):
    if class_fs is None:
        class_fs = [50, 1, 2.5, 5, 10]
    fake_signal = []
    epoch_length = sampling_fs * 30
    for sleep_stage in hypno:
        x = 2 * np.arange(0, epoch_length) * np.pi * class_fs[sleep_stage + 1] / sampling_fs
        x = x + np.random.normal(scale=2)
        x = np.cos(x)
        x = x + np.random.normal(scale=random_noise, size=x.shape)
        fake_signal += [x]
    return np.concatenate(fake_signal)


def generate_fake_record(record_description, file_name='/tmp/fake.h5', build_hypno=True):
    with h5py.File(file_name, 'w') as fake_h5:
        description = []

        hypno_duration = int(3.4 * 60 * 60 // 30)
        transition_kernel_hypno = np.abs(np.random.normal(size=(5, 5)) + 30 * np.identity(5))
        transition_kernel_hypno = transition_kernel_hypno / np.sum(transition_kernel_hypno, 1,
                                                                   keepdims=True)
        hypno = generate_fake_hypno(transition_kernel_hypno, hypno_duration)

        fake_h5['hypnogram'] = hypno
        for signal in record_description:
            group_name = signal['path'].split('/')[0]
            signal_name = signal['path'].split('/')[1]
            signal_description = {"fs": signal['fs'], "unit": '',
                                  "path": 'signals/' + signal['path'], 'name': signal_name,
                                  'domain': group_name, "default": True}
            data = generate_fake_signal(hypno, signal['fs'])
            fake_h5.create_dataset('signals/' + signal['path'], data=data)
            description += [signal_description]
            fake_h5['signals/' + group_name].attrs.create('fs', signal['fs'])
            fake_h5['signals/' + group_name].attrs.create('unit', b'')

        # Add description
        fake_h5.attrs.create('description', json.dumps(description), dtype=np.dtype('S32768'))

        # Add event (nothing for now)
        fake_h5.create_group('events')
        fake_h5.attrs.create('events_description', json.dumps([]), dtype=np.dtype('S32768'))

        # add duration
        duration = standardize_signals_durations(fake_h5)
        fake_h5.attrs.create('duration', duration)
        if not build_hypno:
            del fake_h5['hypnogram']
    return file_name


def generate_memmaps(n_memmaps, record_description, memmaps_description, erase=True):
    dataset_size = n_memmaps

    # generate fake h5
    if not os.path.exists('/tmp/fake_dataset/'):
        os.makedirs('/tmp/fake_dataset/')
    else:
        shutil.rmtree('/tmp/fake_dataset/')
        os.makedirs('/tmp/fake_dataset/')

    if erase:
        if not os.path.exists('/tmp/fake_memmmaps/'):
            os.makedirs('/tmp/fake_memmmaps/')
        else:
            shutil.rmtree('/tmp/fake_memmmaps/')
            os.makedirs('/tmp/fake_memmmaps/')

    for i in range(dataset_size):
        generate_fake_record(record_description, '/tmp/fake_dataset/record_' + str(i) + '.h5')

    # generate fake_memmaps
    records = ['/tmp/fake_dataset/record_' + str(i) + '.h5' for i in range(dataset_size)]
    memmaps_dir = h5_to_memmaps(
        records,
        '/tmp/fake_memmmaps/',
        memmaps_description,
        parallel=False,
        force=True)[0]
    shutil.rmtree('/tmp/fake_dataset/')

    # build dataset
    memmaps = [os.path.join(memmaps_dir, filename) for filename in os.listdir(memmaps_dir) if
               '.' not in filename]
    return memmaps
