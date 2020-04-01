import hashlib
import json
import os

from dreem_learning_open.settings import EXPERIMENTS_DIRECTORY
from dreem_learning_open.settings import SLEEP_EDF_SETTINGS
from dreem_learning_open.utils.run_experiments import run_experiments
from dreem_learning_open.settings import EXPERIMENTS_DIRECTORY, RESULTS_DIRECTORY
import h5py


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


datasets = {'sleep_edf': SLEEP_EDF_SETTINGS}
experiments_directory = 'scripts/sleep_edf/'
mode = 'Extended'

records = os.listdir(SLEEP_EDF_SETTINGS['h5_directory'])
records_for_subject = {}
for record in records:
    subject_id = record[:5]
    if subject_id in records_for_subject:
        records_for_subject[subject_id] += [record.replace('.h5','')]
    else:
        records_for_subject[subject_id] = [record.replace('.h5','')]

if mode == 'SC-20':
    subjects = list(records_for_subject.keys())
    subjects.sort()
    subjects = subjects[:20]
    subjects_split = [{'subject':k,'records':v} for k,v in records_for_subject.items() if k in
                      subjects]
    split = {
        'type': 'kfolds',
        'args': {
            'n_folds': 20,
            'subjects': subjects_split
        }
    }
elif mode == 'extended':
    subjects_split = [{'subject': k, 'records': v} for k, v in records_for_subject.items()]
    split = {
        'type': 'kfolds',
        'args': {
            'n_folds': 10,
            'subjects': subjects_split
        }
    }



experiments = ['deep_sleep_net','seq_sleep_net','mixed_neural_network','chambon_et_al',
               'tsinallis_et_al']
run_experiments(experiments,
                experiments_directory,
                EXPERIMENTS_DIRECTORY,
                split = split,
                datasets=datasets, error_tolerant=False)
