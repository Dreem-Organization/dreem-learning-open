from dreem_learning_open.settings import DODO_SETTINGS
from dreem_learning_open.logger.logger import inference_on_dataset
import json
from dreem_learning_open.preprocessings.h5_to_memmap import h5_to_memmaps
import os

experiment_folder = 'pretrained_model/dodo/simple_sleep_net/'
memmaps_description = json.load(open(f"{experiment_folder}/description.json"))[
    'memmap_description']  # As defined above
records = [DODO_SETTINGS['h5_directory'] + '/' + record for record in
           os.listdir(DODO_SETTINGS['h5_directory'])]
output_memmap_folder, groups_description, features_description = h5_to_memmaps(records,
                                                                               DODO_SETTINGS[
                                                                                   'memmap_directory'],
                                                                               memmaps_description,
                                                                               parallel=False,
                                                                               remove_hypnogram=True)
records_to_eval = [output_memmap_folder + '/' + record + '/' for record in
                   os.listdir(output_memmap_folder) if '.' not in record]
proba = inference_on_dataset(records_to_eval, experiment_folder, return_prob=True)
hypnogram = results = inference_on_dataset(records_to_eval, experiment_folder, return_prob=False)
