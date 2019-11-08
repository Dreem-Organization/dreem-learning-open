import os
import shutil

import numpy as np
import pytest
import torch

from ..datasets.dataset import DreemDataset
from .descriptions import three_groups_record_description, memmaps_description_nested, \
    expected_properties, \
    augmentation_pipeline_nested, \
    memmaps_description_bis, augmentation_pipeline_nested_wrong, groups_description
from .utils import generate_memmaps


def test_dataset_base():
    """
    Test that the dataset hase:
    - the right numbers of record
    - with the right start and end idx
    - with the right shape
    - that the produced hypnogram are valids
    """
    groups = groups_description
    dataset_size = 2
    temporal_context = 3
    try:
        shutil.rmtree('/tmp/fake_memmmaps/')
    except:
        pass
    # build dataset
    memmaps = generate_memmaps(dataset_size, three_groups_record_description,
                               memmaps_description_nested)
    fake_dataset = DreemDataset(groups, features_description={}, records=memmaps,
                                temporal_context=temporal_context)

    ## Basic test on shape

    # All the record have
    assert len(fake_dataset.records) == dataset_size, 'Not the right number of records'

    for group in groups:
        expected_shape = (
            temporal_context, expected_properties[group]['shape'][1],
            30 * expected_properties[group]['fs'])
        assert fake_dataset[0]['groups'][
                   group].size() == expected_shape, 'Group do not have the right shape'

    assert torch.max(fake_dataset[0]['hypnogram']) >= 0, 'Some hypnogram have no valid values'
    assert fake_dataset[0]['hypnogram'].size() == (
    temporal_context,), 'The hypnogram does not have the right shape'

    for record in fake_dataset.record_index:
        idx_begin, idx_end = fake_dataset.record_index[record][0], \
                             fake_dataset.record_index[record][1]
        record_length = int(
            np.sum(fake_dataset.hypnogram[record][
                   temporal_context // 2 + 1:-temporal_context // 2 - 1] > - 1))

        assert idx_end - idx_begin + 1 == record_length, 'Start and end index of the record is wrong'

        current_record_hypnogram = np.memmap(record + 'hypno.mm', mode='r', dtype='float32')
        current_record_signals = os.listdir(record + 'signals/')
        signals_memmap = {signal.replace('.mm', ''): record + 'signals/' + signal for signal in
                          current_record_signals}
        signals_memmap = {
            group_name: np.memmap(group_path, dtype='float32', mode='r',
                                  shape=tuple(expected_properties[group_name]['shape'])) for
            group_name, group_path in signals_memmap.items()}
        valid_epoch = 0

        # validation on the record content
        for i in range(current_record_hypnogram.shape[0]):
            ## assert that the output hypnogram is what we expect
            if current_record_hypnogram[i] != -1:
                # if np.random.uniform()<0.05:
                hypno_dataset = fake_dataset[idx_begin + valid_epoch][
                    'hypnogram'].cpu().float().numpy()
                hypno_memmap = current_record_hypnogram[
                               i - temporal_context // 2:i + temporal_context // 2 + 1]
                assert (hypno_dataset == hypno_memmap).all(), 'The value in the hypnogram are wrong'

                ## check that the input is what we want too:

                for group in groups:
                    X_dataset = fake_dataset[idx_begin + valid_epoch]['groups'][
                        group].cpu().float().numpy()
                    X_dataset = X_dataset.transpose(0, 2, 1).reshape(-1, X_dataset.shape[1])
                    epoch_length = expected_properties[group]['fs'] * 30
                    X_memmap = signals_memmap[group][
                               epoch_length * (i - temporal_context // 2):epoch_length * (
                                       i + temporal_context // 2 + 1)]
                    assert (X_dataset == X_memmap).all(), 'The value in the dataset are wrong'

                valid_epoch += 1

    shutil.rmtree('/tmp/fake_memmmaps/')


def test_add_record():
    """

    """
    groups = groups_description
    dataset_size = 3
    temporal_context = 3
    try:
        shutil.rmtree('/tmp/fake_memmmaps/')
    except:
        pass
    # build dataset
    memmaps = generate_memmaps(dataset_size, three_groups_record_description,
                               memmaps_description_nested)
    fake_dataset = DreemDataset(groups, features_description={}, temporal_context=temporal_context)
    for i, memmap in enumerate(memmaps):
        fake_dataset.add_record(memmap)
        assert len(fake_dataset.records) == i + 1, 'Wrong nummber of records in the dataset'
        assert fake_dataset.records[-1] == memmap, 'The wrong record has been added'
    try:
        shutil.rmtree('/tmp/fake_memmmaps/')
    except:
        pass


def test_get_record():
    """
    Compare the data from dataset.get_record and an iterator built by hand to ensure they are identical
    """
    groups = groups_description
    dataset_size = 3
    temporal_context = 3

    try:
        shutil.rmtree('/tmp/fake_memmmaps/')
    except:
        pass
    # build dataset
    memmaps = generate_memmaps(dataset_size, three_groups_record_description,
                               memmaps_description_nested)
    fake_dataset = DreemDataset(groups, features_description={}, records=memmaps,
                                temporal_context=temporal_context)
    for record in fake_dataset.records:

        # load the file of the current record
        current_record_hypnogram = np.memmap(record + 'hypno.mm', mode='r', dtype='float32')
        current_record_signals = os.listdir(record + 'signals/')
        signals_memmap = {signal.replace('.mm', ''): record + 'signals/' + signal for signal in
                          current_record_signals}
        signals_memmap = {
            group_name: np.memmap(group_path, dtype='float32', mode='r',
                                  shape=tuple(expected_properties[group_name]['shape'])) for
            group_name, group_path in signals_memmap.items()}

        get_record = fake_dataset.get_record(record, 16)
        valid_epoch = 0

        batch_by_hand = {'hypnogram': []}
        for group in groups:
            batch_by_hand[group] = []
        for i in range(current_record_hypnogram.shape[0]):
            if current_record_hypnogram[i] != -1:

                # Built th batch by hand
                batch_by_hand['hypnogram'] += [np.expand_dims(
                    current_record_hypnogram[
                    i - temporal_context // 2:i + temporal_context // 2 + 1], 0)]

                # check that the input is what we want too:

                for group in groups:
                    epoch_length = expected_properties[group]['fs'] * 30
                    X_memmap = signals_memmap[group][
                               epoch_length * (i - temporal_context // 2):epoch_length * (
                                       i + temporal_context // 2 + 1)]
                    X_memmap = np.expand_dims(X_memmap, axis=0)
                    X_memmap = X_memmap.reshape((1, temporal_context, epoch_length, -1)).transpose(
                        (0, 1, 3, 2))

                    batch_by_hand[group] += [X_memmap]

                valid_epoch += 1

                # Once enough record are collected compare with the result from get_record
                if valid_epoch % 16 == 0 and valid_epoch > 0:
                    batch_dataset = get_record.__next__()
                    for _key in batch_by_hand:
                        batch_by_hand[_key] = np.concatenate(batch_by_hand[_key], 0)
                        if _key == 'hypnogram':
                            batch_by_hand[_key] = batch_by_hand[_key]
                            assert np.linalg.norm(
                                batch_by_hand[_key] - batch_dataset[_key].cpu().numpy()) < 1e-12
                        else:
                            assert np.linalg.norm(
                                batch_by_hand[_key] - batch_dataset['groups'][
                                    _key].cpu().numpy()) < 1e-12

                    batch_by_hand = {'hypnogram': []}
                    for group in groups:
                        batch_by_hand[group] = []

        # last batch may not be complete
        try:
            batch_dataset = get_record.__next__()
            for _key in batch_by_hand:

                batch_by_hand[_key] = np.concatenate(batch_by_hand[_key], 0)
                if _key == 'hypnogram':
                    batch_by_hand[_key] = batch_by_hand[_key]
                    assert np.linalg.norm(
                        batch_by_hand[_key] - batch_dataset[_key].cpu().numpy()) < 1e-12
                else:
                    assert np.linalg.norm(
                        batch_by_hand[_key] - batch_dataset['groups'][_key].cpu().numpy()) < 1e-12
        except StopIteration:
            pass


try:
    shutil.rmtree('/tmp/fake_memmmaps/')
except:
    pass


def test_augmentation_pipeline_dataset():
    """
    Test that the augmentation pipeline augment the data
    """
    try:
        shutil.rmtree('/tmp/fake_memmmaps/')
    except:
        pass

    groups = groups_description
    dataset_size = 1
    temporal_context = 3

    # build dataset
    memmaps = generate_memmaps(dataset_size, three_groups_record_description,
                               memmaps_description_nested)
    fake_dataset_with_transform = DreemDataset(groups, features_description={}, records=memmaps,
                                               temporal_context=temporal_context,
                                               transform_parameters=augmentation_pipeline_nested)
    fake_dataset = DreemDataset(groups, features_description={}, records=memmaps,
                                temporal_context=temporal_context)

    # assert over different index
    for i, _ in enumerate(fake_dataset_with_transform):
        assert torch.norm(
            fake_dataset_with_transform[i]['groups']['eeg-eog'] - fake_dataset[i]['groups'][
                'eeg-eog']) > 1e-4

    try:
        shutil.rmtree('/tmp/fake_memmmaps/')
    except:
        pass


def test_different_memmaps_dataset():
    """
    Check that feeding memmaps with different errors will raise an error
    """
    groups = groups_description
    dataset_size = 1
    temporal_context = 3
    try:
        shutil.rmtree('/tmp/fake_memmmaps/')
    except:
        pass

    # build dataset
    memmaps_1 = generate_memmaps(dataset_size, three_groups_record_description,
                                 memmaps_description_nested)
    memmaps_2 = generate_memmaps(dataset_size, three_groups_record_description,
                                 memmaps_description_bis, erase=False)
    with pytest.raises(AssertionError, match='Invalid group shape'):
        dataset = DreemDataset(groups, features_description={}, records=memmaps_1 + memmaps_2,
                               temporal_context=temporal_context)

    try:
        shutil.rmtree('/tmp/fake_memmmaps/')
    except:
        pass


def test_wrong_augmentation_pipeline():
    """
    Test that the augmentation pipeline augment the data
    """
    groups = groups_description
    dataset_size = 1
    temporal_context = 3
    try:
        shutil.rmtree('/tmp/fake_memmmaps/')
    except:
        pass
    # build dataset
    memmaps = generate_memmaps(dataset_size, three_groups_record_description,
                               memmaps_description_nested)

    with pytest.raises(AssertionError, match='augmentation pipeline is invalid'):
        DreemDataset(groups, features_description={}, records=memmaps,
                     temporal_context=temporal_context,
                     transform_parameters=augmentation_pipeline_nested_wrong)
    try:
        shutil.rmtree('/tmp/fake_memmmaps/')
    except:
        pass
