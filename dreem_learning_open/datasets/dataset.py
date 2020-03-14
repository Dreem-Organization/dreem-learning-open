"""
dataset
"""
import copy
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_augmentation import augment_data
from ..utils.utils import get_group_description_from_record_description


class DreemDataset(Dataset):

    def __init__(self, groups_description, features_description=None, transform_parameters=None,
                 temporal_context=1, temporal_context_mode='sequential',
                 records=None):
        if not isinstance(transform_parameters, list) and transform_parameters is not None:
            raise TypeError('transform_parameters should be a list')

        self.hypnogram = {}
        self.data = {}
        self.features_data = {}
        self.idx_to_record = []
        self.idx_to_record_eval = []

        self.temporal_context_mode = temporal_context_mode
        if self.temporal_context_mode == 'sequential':
            self.temporal_context = temporal_context
            self.max_temporal_context = temporal_context
            self.input_temporal_dimension = 1
        elif self.temporal_context_mode == 'concatenated':
            self.temporal_context = 1
            self.max_temporal_context = 1
            self.input_temporal_dimension = temporal_context
        else:
            raise ValueError('temporal_context_mode should be in ["concatenated","sequential"]')

        self.records = []
        self.groups = list(groups_description.keys())
        self.groups_description = copy.deepcopy(groups_description)

        self.features = list(features_description.keys())
        self.features_description = features_description
        self.transform_parameters = transform_parameters
        self.record_index = {}
        self.record_index_eval = {}
        if self.transform_parameters is not None:
            for group in self.groups:
                assert group in [group['name'] for group in
                                 transform_parameters], 'augmentation pipeline is invalid'

        if records is not None:
            for record in records:
                self.add_record(record)

    def get_record(self, record, batch_size=64, return_index=False, mode='train', stride=1):
        if mode == 'train':
            index_min, index_max = self.record_index[record]
        else:
            index_min, index_max = self.record_index_eval[record]
        number_of_samples = index_max - index_min
        for i in range(number_of_samples // batch_size + 1):
            element_in_batch = 0
            batch_data = {}
            batch_data['groups'] = {group: [] for group in self.groups}

            batch_data['features'] = {feature: [] for feature in self.features}
            batch_data['hypnogram'] = []
            indexes = []
            for j in range(i * batch_size * stride, (i + 1) * batch_size * stride, stride):
                if j + index_min <= index_max:
                    data = self.__getitem__(j + index_min, mode=mode)
                    batch_data['hypnogram'] += [data['hypnogram'].unsqueeze(0)]
                    for group in batch_data['groups']:
                        batch_data['groups'][group] += [data['groups'][group].unsqueeze(0)]

                    for feature in batch_data['features']:
                        batch_data['features'][feature] += [data['features'][feature].unsqueeze(0)]
                    element_in_batch += 1
                    if mode == 'train':
                        indexes += [self.idx_to_record[j + index_min]["index"]]
                    else:
                        indexes += [self.idx_to_record_eval[j + index_min]["index"]]

            if element_in_batch > 0:
                for group in batch_data['groups']:
                    batch_data['groups'][group] = torch.cat(batch_data['groups'][group])
                for feature in batch_data['features']:
                    batch_data['features'][feature] = torch.cat(batch_data['features'][feature])

                batch_data['hypnogram'] = torch.cat(batch_data['hypnogram'])
                if return_index:
                    yield batch_data, indexes
                else:
                    yield batch_data

    def add_record(self, record):
        print('Adding: ', record)
        with open(os.path.join(record, 'properties.json')) as f:
            record_description = json.load(f)
        with open(os.path.join(record, 'features_description.json')) as f:
            features_description = json.load(f)

        if len(self.groups_description) > 0:
            assert all(
                self.groups_description[key] == get_group_description_from_record_description(
                    record_description)[key] for key in
                self.groups_description), 'Invalid group shape for' + record

        window_length = max(self.temporal_context, self.input_temporal_dimension)
        self.records += [record]
        self.data[record] = {}
        self.features_data[record] = {}

        groups = record_description
        for group in groups:
            shape = tuple(groups[group]['shape'])
            self.data[record][group] = np.memmap(os.path.join(record, 'signals', group + '.mm'), mode='r',
                                                 dtype='float32',
                                                 shape=shape)

        for feature in self.features_description:
            shape = tuple(features_description[feature]['shape'])
            self.features_data[record][feature] = np.memmap(os.path.join(record, 'features', feature + '.mm'),
                                                            mode='r',
                                                            dtype='float32',
                                                            shape=shape)

        self.hypnogram[record] = np.memmap(os.path.join(record, 'hypno.mm'), mode='r', dtype='float32')

        # Compute window for training
        valid_window = np.where(self.hypnogram[record] != -1)[0]

        valid_window = valid_window[valid_window >= window_length // 2]
        valid_window = valid_window[
            valid_window <= self.hypnogram[record].shape[0] - window_length // 2 - 1]
        self.idx_to_record += [
            {
                "record": record,
                "index": index,
            } for index in valid_window
        ]

        self.record_index[record] = [i for i, idx_to_record in enumerate(self.idx_to_record) if
                                     idx_to_record['record'] == record]
        try:
            self.record_index[record] = (
                np.nanmin(self.record_index[record]), np.nanmax(self.record_index[record]))
        except:
            self.record_index[record] = (np.nan, np.nan)

        # Compute window for evaluation (we never want to predict -1)
        valid_window_eval = np.arange(0, len(self.hypnogram[record]))
        valid_window_eval = valid_window_eval[valid_window_eval >= window_length // 2]
        valid_window_eval = valid_window_eval[
            valid_window_eval <= self.hypnogram[record].shape[0] - window_length // 2 - 1]
        self.idx_to_record_eval += [
            {
                "record": record,
                "index": index,
            } for index in valid_window_eval
        ]

        self.record_index_eval[record] = [i for i, idx_to_record in
                                          enumerate(self.idx_to_record_eval) if
                                          idx_to_record['record'] == record]
        self.record_index_eval[record] = (
            np.min(self.record_index_eval[record]), np.max(self.record_index_eval[record]))

    def set_device(self, device):
        self.device = device

    def __len__(self):
        return len(self.idx_to_record) - 1

    def __getitem__(self, idx, mode='train'):
        sample = {}
        if mode == 'train':
            record = self.idx_to_record[idx]["record"]
            idx = self.idx_to_record[idx]["index"]
            temporal_context = self.temporal_context
        elif mode == 'eval':
            record = self.idx_to_record_eval[idx]["record"]
            idx = self.idx_to_record_eval[idx]["index"]
            temporal_context = self.max_temporal_context
        else:
            raise ValueError

        sample['record'] = record
        sample['groups'] = {}
        sample['features'] = {}

        # retrieve groups
        start_idx = idx - temporal_context // 2
        end_idx = idx + temporal_context // 2 + 1
        for group in self.groups:
            window_length = self.groups_description[group]['window_length']
            input_temporal_length = self.groups_description[group]['window_length'] * (
                self.input_temporal_dimension // 2)
            sample['groups'][group] = self.data[record][group][
                start_idx * window_length - input_temporal_length:end_idx * window_length + input_temporal_length]
            if self.input_temporal_dimension == 1:
                sample['groups'][group] = sample['groups'][group].reshape(temporal_context,
                                                                          window_length, -1)
            else:
                sample['groups'][group] = sample['groups'][group].reshape(temporal_context,
                                                                          window_length * self.input_temporal_dimension,
                                                                          -1)
            sample['groups'][group] = np.transpose(sample['groups'][group], (0, 2, 1))

        # transform
        if self.transform_parameters is not None and mode == 'train':
            sample['groups'] = augment_data(sample['groups'], self.transform_parameters)

        for group in self.groups:
            sample['groups'][group] = torch.Tensor(sample['groups'][group])

        for feature in self.features:
            sample['features'][feature] = torch.Tensor(
                self.features_data[record][feature][start_idx:end_idx])

        # Retrieve hypnogram
        sample['hypnogram'] = torch.LongTensor(
            self.hypnogram[record][start_idx:end_idx]
        )

        return sample

    def serialize(self):
        return {'augmentation_pipeline': self.transform_parameters,
                'temporal_context': self.temporal_context}

    @staticmethod
    def load(serialized_dataset):
        pipeline = serialized_dataset['augmentation_pipeline']
        window_length = serialized_dataset['temporal_context']
        return DreemDataset(pipeline, window_length)
