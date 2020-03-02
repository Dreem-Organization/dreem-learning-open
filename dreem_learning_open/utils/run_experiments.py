import hashlib
import json
import os
import random as rd

from dreem_learning_open.logger.logger import log_experiment
from dreem_learning_open.preprocessings.h5_to_memmap import h5_to_memmaps
from dreem_learning_open.utils.train_test_val_split import train_test_val_split
import shutil


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


def run_experiments(experiments, experiments_directory, output_directory, datasets,
                    fold_to_run=None, force=True, error_tolerant=False):
    for experiment in experiments:
        experiment_directory = os.path.join(experiments_directory, experiment)
        memmaps_description = json.load(open(os.path.join(experiment_directory, 'memmaps.json')))
        for memmap_description in memmaps_description:
            dataset = memmap_description['dataset']
            if dataset in datasets:
                del memmap_description['dataset']
                exp_name = memmap_description.get('name', experiment)
                dataset_parameters = json.load(
                    open(os.path.join(experiment_directory, 'dataset.json')))
                for dataset_parameter in dataset_parameters:
                    if 'name' in dataset_parameter:
                        exp_name_bis = os.path.join(exp_name, dataset_parameter['name'])
                    else:
                        exp_name_bis = exp_name
                    dataset_setting = datasets[dataset]
                    save_folder = os.path.split(output_directory, dataset, exp_name_bis)
                    if os.path.exists(save_folder) and force:
                        shutil.rmtree(save_folder)

                    normalization = json.load(
                        open(os.path.join(experiment_directory, 'normalization.json')))
                    trainer = json.load(open(os.path.join(experiment_directory, 'trainer.json')))
                    transform = json.load(
                        open(os.path.join(experiment_directory, 'transform.json')))
                    net = json.load(open(os.path.join(experiment_directory, 'net.json')))

                    temporal_context = dataset_parameter['temporal_context']
                    temporal_context_mode = dataset_parameter['temporal_context_mode']

                    description_hash = memmap_hash(memmap_description)
                    h5_to_memmaps(
                        records=[os.path.join(dataset_setting['h5_directory'], record) for record in
                                 os.listdir(dataset_setting['h5_directory'])],
                        memmap_description=memmap_description,
                        memmap_directory=dataset_setting['memmap_directory'],
                        parallel=False,
                        error_tolerant=error_tolerant)
                    dataset_dir = os.path.join(
                        dataset_setting['memmap_directory'],
                        description_hash
                    )
                    available_dreem_records = [
                        os.path.join(dataset_dir, record) for record in
                        os.listdir(dataset_dir) if '.json' not in record
                    ]
                    # build the folds
                    rd.seed(2019)
                    rd.shuffle(available_dreem_records)

                    if dataset in ['dodo', 'mass_multi_channel', 'mass']:
                        if dataset == 'dodo':
                            N_FOLDS = 20
                        if dataset in ['mass_multi_channel', 'mass']:
                            N_FOLDS = 31
                        N_FOLDS = N_FOLDS - 1
                        FOLDS_SIZE = int(len(available_dreem_records) // N_FOLDS)
                        folds = [available_dreem_records[FOLDS_SIZE * x:FOLDS_SIZE * (x + 1)] for x
                                 in
                                 range(int(len(available_dreem_records) / FOLDS_SIZE + 1))]

                    else:
                        # LOOV training
                        folds = [[record] for record in available_dreem_records]

                    if fold_to_run is None:
                        fold_to_run = [j for j, _ in enumerate(folds)]

                    for i, fold in enumerate(folds):
                        if i in fold_to_run:
                            other_records = [record for record in available_dreem_records if
                                             record not in fold]
                            rd.seed(2019 + i)
                            rd.shuffle(other_records)
                            train_records, val_records, _ = train_test_val_split(other_records,
                                                                                 0.8, 0.2,
                                                                                 0,
                                                                                 seed=2019)
                            experiment_description = {
                                'memmap_description': memmap_description,
                                'dataset_settings': dataset_setting,
                                'trainer_parameters': trainer,
                                'normalization_parameters': normalization,
                                'net_parameters': net,
                                'dataset_parameters': {
                                    'split': {
                                        'train': train_records,
                                        'val': val_records,
                                        'test': fold
                                    },
                                    'temporal_context': temporal_context,
                                    'temporal_context_mode': temporal_context_mode,
                                    'transform_parameters': transform

                                },
                                'save_folder': os.path.join(output_directory, dataset, exp_name_bis),
                            }

                            log_experiment(**experiment_description, parralel=True,
                                           generate_memmaps=False)
