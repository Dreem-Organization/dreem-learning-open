import os

import pyedflib
import h5py
import pytz
import datetime as dt
import struct

psg_properties = {'digital_max': [32767],
                      'digital_min': [-32767],
                      'dimension': ['uV'],
                      'physical_min': [-800.0],
                      'physical_max': [800.0],
                      'prefilter': [''],
                      'sample_rate': [250],
                      "transducer": [""]}


def convert_h5_to_edf(h5_path, output_file="psg.edf",psg_properties = psg_properties):
    h5 = h5py.File(h5_path, "r")


    # Check that all ?
    subfolders = ['signals/eeg', 'signals/emg', 'signals/eog']
    psg_labels = []
    for subfolder in subfolders:
        psg_labels.extend([f"{subfolder}/{x}" for x in list(h5[subfolder].keys())])

    try:
        start_time = pytz.timezone('UTC').localize(
            dt.datetime.utcfromtimestamp(h5.attrs["start_time"])
        )
    except KeyError:
        start_time = pytz.timezone('UTC').localize(
            dt.datetime.utcfromtimestamp(0)
        )

    number_of_data_records = int(len(h5[psg_labels[0]]) / 250)
    duration = 1
    header = (
        "0".ljust(8)
        + "".ljust(80)
        + "".ljust(80)
        + start_time.strftime("%d.%m.%y%H.%M.%S")
        + str((len(psg_labels) + 1) * 256).ljust(8)
        + "".ljust(44)
        + str(number_of_data_records).ljust(8)
        + str(duration).ljust(8)
        + str(len(psg_labels)).ljust(4)
    )




    subheaders = (
            "".join([str(x.split('/')[-1]).ljust(16) for x in psg_labels])
            + "".join([str(x).ljust(80) for x in psg_properties['transducer'] * len(psg_labels)])
            + "".join([str(x).ljust(8) for x in psg_properties['dimension'] * len(psg_labels)])
            + "".join([str(x).ljust(8) for x in psg_properties['physical_min'] * len(psg_labels)])
            + "".join([str(x).ljust(8) for x in psg_properties['physical_max'] * len(psg_labels)])
            + "".join([str(x).ljust(8) for x in psg_properties['digital_min'] * len(psg_labels)])
            + "".join([str(x).ljust(8) for x in psg_properties['digital_max'] * len(psg_labels)])
            + "".join([str(x).ljust(80) for x in psg_properties['prefilter'] * len(psg_labels)])
            + "".join([str(x).ljust(8) for x in psg_properties['sample_rate'] * len(psg_labels)])
            + "".ljust(32) * len(psg_labels)
    )
    edf_path = output_file

    with open(edf_path, "wb") as f:
        f.write(bytes(header, "UTF-8"))
        f.write(bytes(subheaders, "UTF-8"))

        def transform(x, min, max):
            if max < min:
                min, max = max, min
            x = x.clip(min, max)
            return (((x - min) / (max - min)) * (2 ** 16 - 1) - (2 ** 15)).astype(int)

        data_transformed = []
        for i, data_path in enumerate(psg_labels):
            data_transformed += [transform(h5[data_path][:], psg_properties['physical_min'][0], psg_properties['physical_max'][0])]

        for i in range(number_of_data_records):
            data = []
            for k, signal_transformed in enumerate(data_transformed):
                data += list(signal_transformed[i * int(psg_properties['sample_rate'][0]): int(psg_properties['sample_rate'][0] * (i + 1))])
            data_to_write = struct.pack("h" * len(data), *data)
            f.write(data_to_write)

    return edf_path

