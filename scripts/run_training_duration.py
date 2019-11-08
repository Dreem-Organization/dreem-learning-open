import hashlib
import json
import os
import numpy as np

from dreem_learning_open.logger.logger import log_experiment
from dreem_learning_open.preprocessings.h5_to_memmap import h5_to_memmaps
from dreem_learning_open.settings import DODH_SETTINGS
from dreem_learning_open.settings import EXPERIMENTS_DIRECTORY
from dreem_learning_open.utils.train_test_val_split import train_test_val_split


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


datasets = {'dodh': DODH_SETTINGS}
experiment_name = "training_duration"
experiments_directory = f'scripts/{experiment_name}/'
models = [experiment for experiment in os.listdir(experiments_directory) if os.path.isdir(
    experiments_directory + experiment)]

for model in models:
    experiment_directory = experiments_directory + model + '/'
    memmaps_description = json.load(open(experiment_directory + 'memmaps.json'))
    for memmap_description in memmaps_description:
        dataset = memmap_description['dataset']
        del memmap_description['dataset']
        others = json.load(open(experiment_directory + 'dataset.json'))
        for other in others:
            dataset_setting = datasets[dataset]

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

            train_records, val_records, test_record = train_test_val_split(available_dreem_records,
                                                                           train=0.70,
                                                                           test=0.25,
                                                                           val=0.05,
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
                        'test': test_record,
                    },
                    'temporal_context': temporal_context,
                    'temporal_context_mode': temporal_context_mode,
                    'transform_parameters': transform

                },
                'save_folder': f"{EXPERIMENTS_DIRECTORY}/{experiment_name}/{model}"

            }
            save_folder = log_experiment(**experiment_description,
                                         parralel=True, generate_memmaps=False)

            training_durations = [
                metrics["training_duration"]
                for metrics in
                [json.load(open(save_folder + "/training/" + metrics_path, "r")) for metrics_path in
                 os.listdir(
                     save_folder + "/training/") if "metrics" in metrics_path]
            ]

            print(model, np.mean(training_durations), np.std(training_durations))
