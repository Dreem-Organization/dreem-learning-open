import json
import os
import shutil

import pytest

from ..preprocessings.h5_to_memmap import h5_to_memmaps
from .descriptions import memmaps_description_nested, three_groups_record_description, \
    wrong_frequency_memmaps_description, wrong_padding_memmaps_description, \
    invalid_path_description, \
    invalid_processing_args, invalid_processing_name, expected_properties
from .utils import generate_fake_record


def test_h5_to_memmaps_depth():
    """
    Check that h5_to_memmaps work, yields the right number of files with the corrected properties
    """
    generate_fake_record(three_groups_record_description)
    memmaps_description = memmaps_description_nested
    memmap_directory = \
    h5_to_memmaps(['/tmp/fake.h5'], '/tmp/memmap_test/', memmap_description=memmaps_description,
                  parallel=False)[0]
    assert len(os.listdir(memmap_directory + 'fake/')) == 5
    assert set(os.listdir(memmap_directory + 'fake/signals')) == set(
        [group['name'] + '.mm' for group in memmaps_description['signals']])
    with open(memmap_directory + 'fake/properties.json', 'r') as f:
        assert json.load(f) == expected_properties
    shutil.rmtree('/tmp/memmap_test/')
    os.remove('/tmp/fake.h5')


def test_h5_to_memmaps_no_hypno():
    """
    Check that h5_to_memmaps work, yields the right number of files with the corrected properties
    """
    generate_fake_record(three_groups_record_description, build_hypno=False)
    memmaps_description = memmaps_description_nested
    memmap_directory = \
    h5_to_memmaps(['/tmp/fake.h5'], '/tmp/memmap_test/', memmap_description=memmaps_description,
                  parallel=False)[0]
    assert len(os.listdir(memmap_directory + 'fake/')) == 5
    assert set(os.listdir(memmap_directory + 'fake/signals')) == set(
        [group['name'] + '.mm' for group in memmaps_description['signals']])
    with open(memmap_directory + 'fake/properties.json', 'r') as f:
        assert json.load(f) == expected_properties
    shutil.rmtree('/tmp/memmap_test/')
    os.remove('/tmp/fake.h5')


#
def test_wrong_frequency():
    """
    Check that feeding two signals with different fs in the same group will raise an Error
    """
    generate_fake_record(three_groups_record_description)
    memmaps_description = wrong_frequency_memmaps_description
    with pytest.raises(AssertionError, match='frequencies'):
        h5_to_memmaps(['/tmp/fake.h5'], '/tmp/memmap_test/', memmap_description=memmaps_description,
                      parallel=False)
    shutil.rmtree('/tmp/memmap_test/')
    os.remove('/tmp/fake.h5')


def test_wrong_padding():
    """
    Check that feeding two signals with different padding in the same group will raise an Error
    """
    generate_fake_record(three_groups_record_description)
    memmaps_description = wrong_padding_memmaps_description
    with pytest.raises(AssertionError, match='padding'):
        h5_to_memmaps(['/tmp/fake.h5'], '/tmp/memmap_test/', memmap_description=memmaps_description,
                      parallel=False)
    shutil.rmtree('/tmp/memmap_test/')
    os.remove('/tmp/fake.h5')


#
def test_wrong_description():
    """
    Check that feeding a memmaps description with invalid path will raise an error
    """
    generate_fake_record(three_groups_record_description)
    memmaps_description = invalid_path_description
    with pytest.raises(KeyError):
        h5_to_memmaps(['/tmp/fake.h5'], '/tmp/memmap_test/', memmap_description=memmaps_description,
                      parallel=False)
    shutil.rmtree('/tmp/memmap_test/')
    os.remove('/tmp/fake.h5')


#
def test_wrong_processing_args():
    """
    Check that feeding a memmaps description with wrong processing args raise an Error
    """
    generate_fake_record(three_groups_record_description)
    memmaps_description = invalid_processing_args
    with pytest.raises(TypeError):
        h5_to_memmaps(['/tmp/fake.h5'], '/tmp/memmap_test/', memmap_description=memmaps_description,
                      parallel=False)
    shutil.rmtree('/tmp/memmap_test/')
    os.remove('/tmp/fake.h5')


def test_wrong_processing_name():
    """
    Check that feeding a memmaps description with wrong processing operation raise an Error
    """
    generate_fake_record(three_groups_record_description)
    memmaps_description = invalid_processing_name
    with pytest.raises(KeyError):
        h5_to_memmaps(['/tmp/fake.h5'], '/tmp/memmap_test/', memmap_description=memmaps_description,
                      parallel=False)
    shutil.rmtree('/tmp/memmap_test/')
    os.remove('/tmp/fake.h5')
