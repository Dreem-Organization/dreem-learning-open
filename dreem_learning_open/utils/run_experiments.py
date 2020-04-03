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

def split_list(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def run_experiments(experiments, experiments_directory, output_directory, datasets,
                    split,
                    fold_to_run=None, force=False,error_tolerant = False,force_name =None):
    for experiment in experiments:

        experiment_directory = experiments_directory + experiment + '/'
        memmaps_description = json.load(open(experiment_directory + 'memmaps.json'))
        for memmap_description in memmaps_description:
            dataset = memmap_description['dataset']
            if dataset in datasets:
                del memmap_description['dataset']
                exp_name = memmap_description.get('name', experiment)
                dataset_parameters = json.load(open(experiment_directory + 'dataset.json'))
                for dataset_parameter in dataset_parameters:
                    if force_name is not None:
                        exp_name_bis = f"{force_name}/{exp_name}"
                    elif 'name' in dataset_parameter:
                        exp_name_bis = f"{exp_name}/{dataset_parameter['name']}"
                    else:
                        exp_name_bis = exp_name
                    dataset_setting = datasets[dataset]
                    save_folder = f"{output_directory}/{dataset}/{exp_name_bis}/"
                    if os.path.exists(save_folder) and force:
                        shutil.rmtree(save_folder)

                    normalization = json.load(open(experiment_directory + 'normalization.json'))
                    trainer = json.load(open(experiment_directory + 'trainer.json'))
                    transform = json.load(open(experiment_directory + 'transform.json'))
                    net = json.load(open(experiment_directory + 'net.json'))

                    temporal_context = dataset_parameter['temporal_context']
                    temporal_context_mode = dataset_parameter['temporal_context_mode']

                    description_hash = memmap_hash(memmap_description)
                    h5_to_memmaps(
                        records=[dataset_setting['h5_directory'] + record for record in
                                 os.listdir(dataset_setting['h5_directory'])],
                        memmap_description=memmap_description,
                        memmap_directory=dataset_setting['memmap_directory'],
                        parallel=False,error_tolerant=error_tolerant)
                    dataset_dir = dataset_setting['memmap_directory'] + description_hash + '/'
                    available_dreem_records = [dataset_dir + record + '/' for record in
                                               os.listdir(dataset_dir) if '.json' not in record]

                    # build the folds
                    rd.seed(2019)
                    rd.shuffle(available_dreem_records)

                    assert split['type'] in ['loov','kfolds']

                    if split['type'] == 'kfolds':
                        N_FOLDS = split['args']['n_folds']
                        if 'subjects' not in split['args']: # assumer record-wise split
                            folds = split_list(available_dreem_records,N_FOLDS)
                        else: # assume multiple record per subject and subject-wise split
                            subjects = []
                            for subject in split['args']['subjects']:
                                for record in subject['records']:
                                    if record in os.listdir(dataset_dir):
                                        subjects += [subject]
                                        break

                            subject_per_folds = split_list(subjects,N_FOLDS)

                            folds = []
                            for subjects in subject_per_folds:
                                record_in_fold = []
                                for subject in subjects:
                                    for record in subject['records']:
                                        record_in_fold += [dataset_dir + record + '/']
                                folds += [record_in_fold]

                    elif split['type'] == 'loov':
                        # LOOV training
                        folds = [[record] for record in available_dreem_records]
                    else:
                        raise ValueError

                    if fold_to_run is None:
                        fold_to_run = [j for j, _ in enumerate(folds)]

                    for i, fold in enumerate(folds):

                        if i in fold_to_run:
                            other_folds = [fold for k,fold in enumerate(folds) if k!= i]
                            print(len(other_folds))
                            rd.seed(2019 + i)
                            rd.shuffle(other_folds)
                            n_val = max(1,int(len(other_folds) * 0.2))
                            train_folds, val_folds = other_folds[n_val:],other_folds[:n_val]
                            train_records = [record for train_fold in
                                             train_folds for record in train_fold ]
                            val_records = [record for val_fold in val_folds for record in val_fold ]
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
                                'save_folder': f"{output_directory}/{dataset}/{exp_name_bis}/"

                            }

                            log_experiment(**experiment_description, parralel=True,
                                           generate_memmaps=False)
