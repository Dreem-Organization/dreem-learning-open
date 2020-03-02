import os

VERSION = "1_00"
# Suggested directory where H5 and memmaps will be stored
BASE_DIRECTORY = "/data/"
BASE_DIRECTORY_H5 = BASE_DIRECTORY + "h5/"
BASE_DIRECTORY_MEMMAP = BASE_DIRECTORY + "memmap/"
EXPERIMENTS_DIRECTORY = BASE_DIRECTORY + 'experiments/'
RESULTS_DIRECTORY = BASE_DIRECTORY + 'results/'

if not os.path.isdir(BASE_DIRECTORY):
    os.mkdir(BASE_DIRECTORY)

if not os.path.isdir(BASE_DIRECTORY_H5):
    os.mkdir(BASE_DIRECTORY_H5)

if not os.path.isdir(BASE_DIRECTORY_MEMMAP):
    os.mkdir(BASE_DIRECTORY_MEMMAP)

if not os.path.isdir(EXPERIMENTS_DIRECTORY):
    os.mkdir(EXPERIMENTS_DIRECTORY)

if not os.path.isdir(RESULTS_DIRECTORY):
    os.mkdir(RESULTS_DIRECTORY)

DODH_SETTINGS = {
    'h5_directory': BASE_DIRECTORY_H5 + 'dodh/',
    'memmap_directory': BASE_DIRECTORY_MEMMAP + 'dodh/'
}

DODO_SETTINGS = {
    'h5_directory': BASE_DIRECTORY_H5 + 'dodo/',
    'memmap_directory': BASE_DIRECTORY_MEMMAP + 'dodo/'
}

folders_to_create = [
    DODH_SETTINGS['h5_directory'],
    DODH_SETTINGS['memmap_directory'],
    DODO_SETTINGS['h5_directory'],
    DODO_SETTINGS['memmap_directory']
]
MASS_SETTINGS = {
    'h5_directory': BASE_DIRECTORY_H5 + 'mass/',
    'memmap_directory': BASE_DIRECTORY_MEMMAP + 'mass/'
}
folders_to_create = [
    DODH_SETTINGS['h5_directory'],
    DODH_SETTINGS['memmap_directory'],
    DODO_SETTINGS['h5_directory'],
    DODO_SETTINGS['memmap_directory']
]
folders_to_create += [v for v in MASS_SETTINGS.values()]

for folder in folders_to_create:
    if not os.path.exists(folder):
        os.mkdir(folder)
