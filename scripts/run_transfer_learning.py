import hashlib
import json
import os
import random as rd

from dreem_learning_open.logger.logger import log_experiment
from dreem_learning_open.preprocessings.h5_to_memmap import h5_to_memmaps
from dreem_learning_open.settings import EXPERIMENTS_DIRECTORY
from dreem_learning_open.settings import DODO_SETTINGS, DODH_SETTINGS

TRIALS = 20


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


datasets = {'dodh': DODH_SETTINGS, 'dodo': DODO_SETTINGS}
experiments_directory = 'scripts/direct_transfer_learning/'
experiments = [experiment for experiment in os.listdir(experiments_directory) if os.path.isdir(
    experiments_directory + experiment)]
memmaps_hash = {}

for _ in range(TRIALS):
    for experiment in experiments:
        experiment_directory = experiments_directory + experiment + '/'
        memmaps_description = json.load(open(experiment_directory + 'memmaps.json'))
        for memmap_description in memmaps_description:
            dataset = memmap_description['dataset']
            memmaps_hash[dataset] = memmap_hash(memmap_description)
            h5_to_memmaps(
                records=[datasets[dataset]['h5_directory'] + record for record in
                         os.listdir(datasets[dataset]['h5_directory'])],
                memmap_description=memmap_description,
                memmap_directory=datasets[dataset]['memmap_directory'],
                parallel=False)

        for i, source_memmap in enumerate(memmaps_description):
            for j, target_memmap in enumerate(memmaps_description):
                if i != j:
                    source_dataset = source_memmap['dataset']
                    source_dataset_setting = datasets[source_dataset]
                    target_dataset = target_memmap['dataset']
                    target_dataset_setting = datasets[target_dataset]

                    exp_name = f"{experiment}_{source_dataset}_to_{target_dataset}"
                    other = json.load(open(experiment_directory + 'dataset.json'))[0]

                    normalization = json.load(open(experiment_directory + 'normalization.json'))
                    trainer = json.load(open(experiment_directory + 'trainer.json'))
                    transform = json.load(open(experiment_directory + 'transform.json'))
                    net = json.load(open(experiment_directory + 'net.json'))

                    temporal_context, input_temporal_context = other['temporal_context'], other[
                        'input_temporal_context']

                    source_dataset_dir = source_dataset_setting['memmap_directory'] + memmaps_hash[
                        source_dataset] + '/'
                    source_records = [source_dataset_dir + record + '/' for record in
                                      os.listdir(source_dataset_dir)
                                      if
                                      '.json' not in record]
                    target_dataset_dir = target_dataset_setting['memmap_directory'] + memmaps_hash[
                        target_dataset] + '/'
                    target_records = [target_dataset_dir + record + '/' for record in
                                      os.listdir(target_dataset_dir)
                                      if
                                      '.json' not in record]

                    rd.seed(2019)
                    train_prop = 0.7
                    rd.shuffle(source_records)
                    train_records, val_records = source_records[:int(
                        len(source_records) * train_prop)], source_records[int(
                        len(source_records) * train_prop):]

                    # build the folds
                    experiment_description = {
                        'memmap_description': source_memmap,
                        'dataset_settings': source_dataset_setting,
                        'trainer_parameters': trainer,
                        'normalization_parameters': normalization,
                        'net_parameters': net,
                        'dataset_parameters': {
                            'split': {
                                'train': train_records,
                                'val': val_records,
                                'test': target_records
                            },
                            'temporal_context': temporal_context,
                            'input_temporal_dimension': input_temporal_context,
                            'transform_parameters': transform

                        },
                        'save_folder': f"{EXPERIMENTS_DIRECTORY}/{exp_name}/"

                    }

                    folder = log_experiment(**experiment_description, parralel=True,
                                            generate_memmaps=False)

experiments_folder = {
    'dodo_to_dodh': f"{EXPERIMENTS_DIRECTORY}dodo_to_dodh/",
    'dodh_to_dodo': f"{EXPERIMENTS_DIRECTORY}dodh_to_dodo/"
}

for dataset in experiments_folder:
    algo_names = os.listdir(experiments_folder[dataset])
    # Build hypnograms from results file
    for algo_name in algo_names:
        directories_with_experiments = experiments_folder[dataset] + algo_name + '/'
        trials = os.listdir(directories_with_experiments)
        for trial in trials:
            hypnograms = json.load(
                open(directories_with_experiments + trial + '/hypnograms.json', 'r'))
            for dodh_id, hypnogram in hypnograms.items():

                if not os.path.exists(
                        f"results/transfer_learning/{dataset}/{algo_name}/{trial}/"):
                    os.makedirs(
                        f"results/transfer_learning/{dataset}/{algo_name}/{trial}/")

                    with open(f"results/transfer_learning/{dataset}/" +
                              f"{algo_name}/{trial}/{dodh_id}.json", 'w') as outfile:
                        json.dump(hypnogram, outfile, indent=4)
