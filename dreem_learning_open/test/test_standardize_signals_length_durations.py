import h5py

from .descriptions import three_groups_record_description
from .utils import generate_fake_record


def test_standardize_signals_length_durations():
    record_description = three_groups_record_description
    generate_fake_record(record_description)
    with h5py.File('/tmp/fake.h5', 'r') as fake_h5:
        record_duration = fake_h5.attrs['duration']
        assert len(fake_h5['hypnogram'][:]) == record_duration // 30, print(
            len(fake_h5['hypnogram'][:]),
            record_duration // 30)
        for group_name, group in fake_h5['signals'].items():
            fs = group.attrs['fs']
            for signal in group:
                assert len(group[signal]) / fs == record_duration
