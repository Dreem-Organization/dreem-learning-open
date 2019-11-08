import hashlib
import json
import os

from dreem_learning_open.settings import DODH_SETTINGS
from dreem_learning_open.settings import EXPERIMENTS_DIRECTORY
from dreem_learning_open.utils.run_experiments import run_experiments


def memmap_hash(memmap_description):
    return hashlib.sha1(json.dumps(memmap_description).encode()).hexdigest()[:10]


datasets = {'dodh': DODH_SETTINGS}
experiments_directory = 'scripts/base_experiments/'
run_experiments(['chambon_et_al'], experiments_directory, EXPERIMENTS_DIRECTORY,
                datasets=datasets, fold_to_run=list(range(14, 25)))
run_experiments(['tsinalis_et_al', 'deep_sleep_net'], experiments_directory, EXPERIMENTS_DIRECTORY,
                datasets=datasets)

# format json for dod evaluation
experiments_folder = {
    'dodo': f"{EXPERIMENTS_DIRECTORY}dodo/",
    'dodh': f"{EXPERIMENTS_DIRECTORY}dodh/"
}
table = 'base_experiments'
for dataset in datasets:
    algo_names = os.listdir(experiments_folder[dataset])
    for algo_name in algo_names:
        directories_with_experiments = experiments_folder[dataset] + algo_name + '/'
        records = os.listdir(directories_with_experiments)
        for record in records:
            hypnograms = json.load(
                open(directories_with_experiments + record + '/hypnograms.json', 'r'))
            for dodh_id, hypnogram in hypnograms.items():

                if not os.path.exists(
                        'results/{}/{}/{}/'.format(dataset, table, algo_name)):
                    os.makedirs(
                        'results/{}/{}/{}/'.format(dataset, table, algo_name))

                with open('results/{}/{}/{}/'.format(
                        dataset, table, algo_name) + dodh_id + '.json', 'w') as outfile:
                    json.dump(hypnogram, outfile, indent=4)
