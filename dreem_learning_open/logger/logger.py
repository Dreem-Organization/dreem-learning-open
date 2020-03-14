import hashlib
import json
import os
import shutil
import time
import uuid

import git

from ..datasets.dataset import DreemDataset
from ..models.modulo_net.net import ModuloNet
from ..models.modulo_net.normalization import initialize_standardization_parameters
from ..preprocessings.h5_to_memmap import h5_to_memmaps
from ..trainers import Trainer
from ..utils.train_test_val_split import train_test_val_split


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


def log_experiment(dataset_settings, memmap_description, dataset_parameters,
                   normalization_parameters,
                   trainer_parameters,
                   net_parameters=None, save_folder=None, checkpoint=None, parralel=True,
                   experiment_id=None,
                   generate_memmaps=True):
    if experiment_id is None:
        experiment_id = str(uuid.uuid4())

    if net_parameters == checkpoint:
        assert net_parameters is not None, 'Either the net parameters or an experiment to preload have to be provided'

    save_folder = os.path.join(save_folder, experiment_id)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    repo = git.Repo(search_parent_directories=True)
    metadata = {'git_branch': repo.active_branch.name, 'git_hash': repo.head.object.hexsha,
                'begin': int(time.time()), 'end': None, 'experiment_id': experiment_id
                }
    if 'records_name' not in dataset_settings:
        records = [dataset_settings['h5_directory'] + record for record in
                   os.listdir(dataset_settings['h5_directory'])]
    else:
        records = [dataset_settings['h5_directory'] + record for record in
                   dataset_settings['records_name']]

    if generate_memmaps:
        memmaps_directory, groups_description, features_description = h5_to_memmaps(
            records=records,
            memmap_description=memmap_description,
            memmap_directory=dataset_settings[
                'memmap_directory'],
            parallel=parralel)

        memmap_records = [os.path.join(memmaps_directory, record) for record in
                          os.listdir(memmaps_directory) if '.' not in record]

    else:
        memmaps_directory = os.path.join(dataset_settings['memmap_directory'], memmap_hash(
            memmap_description))
        groups_description = json.load(
            open(os.path.join(memmaps_directory, 'groups_description.json'), 'r'))
        features_description = json.load(
            open(os.path.join(memmaps_directory, 'features_description.json'), 'r'))

    if isinstance(dataset_parameters['split']['train'], list):
        train_records, test_records, validation_records = (dataset_parameters['split']['train'],
                                                           dataset_parameters['split']['test'],
                                                           dataset_parameters['split']['val'])
    else:
        train_records, test_records, validation_records = train_test_val_split(memmap_records,
                                                                               **dataset_parameters[
                                                                                   'split'])

    for i, record in enumerate(test_records):
        assert record not in train_records
        assert record not in validation_records

    print('Training records:', len(train_records))
    print('Validation records:', len(validation_records))
    print('Test records:', len(test_records))

    dataset_train = DreemDataset(groups_description, features_description=features_description,
                                 transform_parameters=dataset_parameters['transform_parameters'],
                                 temporal_context=dataset_parameters['temporal_context'],
                                 temporal_context_mode=dataset_parameters['temporal_context_mode'],
                                 records=train_records)
    dataset_validation = DreemDataset(groups_description, features_description=features_description,
                                      temporal_context=dataset_parameters['temporal_context'],
                                      temporal_context_mode=dataset_parameters[
                                          'temporal_context_mode'],
                                      records=validation_records)

    dataset_test = DreemDataset(groups_description, features_description=features_description,
                                temporal_context=dataset_parameters['temporal_context'],
                                temporal_context_mode=dataset_parameters['temporal_context_mode'],
                                records=test_records)

    experiment_description = {
        'metadata': metadata,
        'dataset_settings': dataset_settings,
        'memmap_description': memmap_description,
        'groups_description': groups_description,
        'dataset_parameters': dataset_parameters,

        'normalization_parameters': normalization_parameters,
        'trainers_parameters': trainer_parameters,
        'net_parameters': net_parameters,
        'performance_on_test_set': None,
        'performance_per_records': None,

        'records_split': None,
    }
    json.dump(experiment_description,
              open(os.path.join(save_folder, "description.json"), "w"), indent=4)

    if checkpoint is not None:
        assert 'directory' in checkpoint, 'The directory of the experiment to load has to be provided'
        assert 'net_to_load' in checkpoint, 'The net to load has to be provided'
        shutil.copytree(checkpoint['directory'], os.path.join(save_folder, 'base_experiment'))
        net = ModuloNet.load(
            os.path.join(checkpoint['directory'], checkpoint['net_to_load']))

        if 'trainable_layers' in checkpoint:
            for name, param in net.named_parameters():
                if name not in checkpoint["trainable_layers"]:
                    print("Freezing: ", name)
                    param.requires_grad = False

        with open(os.path.join(checkpoint['directory'], "description.json"), "r") as desc_json:
            net_parameters = json.load(desc_json)['net_parameters']
        print('Load net with parameters:', net_parameters)

    else:
        normalization_parameters_init = initialize_standardization_parameters(
            dataset_train,
            normalization_parameters)
        net = ModuloNet(groups=dataset_train.groups_description,
                        features=dataset_train.features_description,
                        normalization_parameters=normalization_parameters_init,
                        net_parameters=net_parameters)

    trainer_save_folder = os.path.join(save_folder, 'training')
    if not os.path.exists(trainer_save_folder):
        os.makedirs(trainer_save_folder)

    trainer = Trainer(net=net, save_folder=trainer_save_folder,
                      **trainer_parameters['args']
                      )

    trainer.train(train_dataset=dataset_train, validation_dataset=dataset_validation)

    metadata['end'] = int(time.time())

    best_net = ModuloNet.load(os.path.join(trainer_save_folder, 'best_net'))
    trainer = Trainer(net=best_net, save_folder=trainer_save_folder,
                      **trainer_parameters['args'])

    performance_on_test_set, _, performance_per_records, hypnograms = trainer.validate(dataset_test,
                                                                                       return_metrics_per_records=True)
    performance_per_records = {os.path.split(record)[-2]: metric for record, metric in
                               performance_per_records.items()}

    records_split = {
        'train_records': [os.path.split(record)[-2] for record in train_records],
        'validation_records': [os.path.split(record)[-2] for record in validation_records],
        'test_records': [os.path.split(record)[-2] for record in test_records]
    }
    # experiment_description
    experiment_description = {
        'metadata': metadata,
        'dataset_settings': dataset_settings,
        'memmap_description': memmap_description,
        'groups_description': groups_description,
        'features_description': features_description,
        'dataset_parameters': dataset_parameters,

        'normalization_parameters': normalization_parameters,
        'trainers_parameters': trainer_parameters,
        'net_parameters': net_parameters,
        'performance_on_test_set': performance_on_test_set,

        'performance_per_records': performance_per_records,

        'records_split': records_split
    }

    # dump description
    json.dump(experiment_description,
              open(os.path.join(save_folder, "description.json"), "w"), indent=4)

    for group in groups_description:
        padding = groups_description[group]['padding'] // 30

    if padding > 0:
        hypnograms = {k: x['predicted'][padding:-padding] for k, x in hypnograms.items()}
    else:
        hypnograms = {k: x['predicted'] for k, x in hypnograms.items()}

    json.dump(hypnograms,
              open(os.path.join(save_folder, "hypnograms.json"), "w"), indent=4)

    best_net.save(os.path.join(save_folder, 'best_model.gz'))

    return save_folder


def inference_on_dataset(records, experiment_folder, return_prob=False):
    with open(os.path.join(experiment_folder, 'description.json'), 'r') as f:
        experiment_description = json.load(f)
        groups_description = experiment_description['groups_description']
        features_description = experiment_description['features_description']
        for group in groups_description:
            padding = groups_description[group]['padding'] // 30
        dataset_parameters = experiment_description['dataset_parameters']

    net = ModuloNet.load(os.path.split(experiment_folder, 'best_model.gz'))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    print(count_parameters(net))

    dataset_test = DreemDataset(groups_description, features_description=features_description,
                                temporal_context=dataset_parameters['temporal_context'],
                                records=records)
    results = net.predict_on_dataset(dataset_test, verbose=True, return_prob=return_prob)

    if padding > 0:
        results = {k: x[padding:-padding] for k, x in results.items()}
    return results
