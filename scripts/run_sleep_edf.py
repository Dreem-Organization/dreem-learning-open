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

experiments = ['simple_sleep_net']
run_experiments(experiments,
                experiments_directory,
                EXPERIMENTS_DIRECTORY,
                datasets=datasets, error_tolerant=False)
