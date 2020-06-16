import hashlib
import json

from dreem_learning_open.settings import EXPERIMENTS_DIRECTORY
from dreem_learning_open.settings import MASS_SETTINGS
from dreem_learning_open.utils.run_experiments import run_experiments


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


datasets = {'mass': MASS_SETTINGS}
experiments_directory = 'scripts/mass/'
split = {
    'type': 'kfolds',
    'args': {
        "n_folds": 31
    }
}
experiments = ['tsinalis_et_al','chambon_et_al']
run_experiments(experiments,
                experiments_directory,
                EXPERIMENTS_DIRECTORY,
                datasets=datasets, error_tolerant=False, split=split)
