import hashlib
import json
import os
import random as rd

from dreem_learning_open.logger.logger import log_experiment
from dreem_learning_open.preprocessings.h5_to_memmap import h5_to_memmaps
from dreem_learning_open.settings import EXPERIMENTS_DIRECTORY
from dreem_learning_open.settings import DODO_SETTINGS, DODH_SETTINGS


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


datasets = {'dodh': DODH_SETTINGS, 'dodo': DODO_SETTINGS}
dataset_training_settings = {
    'dodh': {'train': [1, 2, 3, 5, 7, 9, 12, 15, 19], 'validation': 3, 'test': 3},
    'dodo': {'train': [1, 2, 3, 5, 7, 9, 12, 15, 19, 25, 30], 'validation': 5, 'test': 5}}

experiments_directory = 'scripts/single_channel/'
experiments = [experiment for experiment in os.listdir(experiments_directory) if os.path.isdir(
    experiments_directory + experiment)]

# LOOV training
for experiment in experiments:
    experiment_directory = experiments_directory + experiment + '/'
    memmaps_description = json.load(open(experiment_directory + 'memmaps.json'))

    for trial_number in range(20):

        for memmap_description in memmaps_description:
            dataset = memmap_description['dataset']
            exp_name = memmap_description.get('name', experiment)
            others = json.load(open(experiment_directory + 'dataset.json'))
            for other in others:
                exp_name = f"{exp_name}/{other['name']}" if 'name' in other else exp_name

                dataset_setting = datasets[dataset]
                dataset_training_setting = dataset_training_settings[dataset]

                normalization = json.load(open(experiment_directory + 'normalization.json'))
                trainer = json.load(open(experiment_directory + 'trainer.json'))
                transform = json.load(open(experiment_directory + 'transform.json'))
                net = json.load(open(experiment_directory + 'net.json'))

                temporal_context, temporal_context_mode = other['temporal_context'], other[
                    'temporal_context_mode']

                description_hash = memmap_hash(memmap_description)
                h5_to_memmaps(
                    records=[dataset_setting['h5_directory'] + record for record in
                             os.listdir(dataset_setting['h5_directory'])],
                    memmap_description=memmap_description,
                    memmap_directory=dataset_setting['memmap_directory'],
                    parallel=False)
                dataset_dir = dataset_setting['memmap_directory'] + description_hash + '/'
                available_dreem_records = [dataset_dir + record + '/' for record in
                                           os.listdir(dataset_dir)
                                           if
                                           '.json' not in record]

                # build the folds
                rd.seed(trial_number)
                rd.shuffle(available_dreem_records)
                validation_size, test_size = dataset_training_setting['validation'], \
                                             dataset_training_setting['test']
                val_set = available_dreem_records[:validation_size]
                test_set = available_dreem_records[
                           validation_size:validation_size + validation_size]
                records = available_dreem_records[validation_size + validation_size:]
                for number_of_training_records in dataset_training_setting['train']:
                    train_set = records[:number_of_training_records]
                    assert len(val_set) == validation_size
                    assert len(test_set) == test_size

                    experiment_description = {
                        'memmap_description': memmap_description,
                        'dataset_settings': dataset_setting,
                        'trainer_parameters': trainer,
                        'normalization_parameters': normalization,
                        'net_parameters': net,
                        'dataset_parameters': {
                            'split': {
                                'train': train_set,
                                'val': val_set,
                                'test': test_set
                            },
                            'temporal_context': temporal_context,
                            'temporal_context_mode': temporal_context_mode,
                            'transform_parameters': transform

                        },
                        'save_folder': f"{EXPERIMENTS_DIRECTORY}/{dataset}/{exp_name}/{number_of_training_records}/"

                    }
                    folder = log_experiment(**experiment_description, parralel=True,
                                            generate_memmaps=False)

experiments_folder = {
    'dodo': f"{EXPERIMENTS_DIRECTORY}learning_curve_dodo/",
    'dodh': f"{EXPERIMENTS_DIRECTORY}learning_curve_dodh/"
}

for dataset in datasets:
    algo_names = os.listdir(experiments_folder[dataset])
    for algo_name in algo_names:
        directories_with_experiments = experiments_folder[dataset] + algo_name + '/'
        trials = os.listdir(directories_with_experiments)
        for trial in trials:
            hypnograms = json.load(
                open(directories_with_experiments + trial + '/hypnograms.json', 'r'))
            for dodh_id, hypnogram in hypnograms.items():

                preds = hypnogram['predicted']

                if not os.path.exists(
                        'results/{}/{}/{}/'.format(dataset, algo_name, trial)):
                    os.makedirs(
                        'results/{}/{}/{}/'.format(dataset, algo_name, trial))

                with open('results/{}/{}/{}/'.format(
                        dataset, algo_name, trial) + dodh_id + '.json', 'w') as outfile:
                    json.dump(preds, outfile, indent=4)
