from copy import deepcopy
from random import seed as set_seed
from random import shuffle


def train_test_val_split(records_list, train=0.6, test=0.2, val=0.2, seed=None):
    assert train + test + val == 1
    split = deepcopy(records_list)
    train_size, test_size, val_size = round(len(split) * train), round(len(split) * test), round(
        len(split) * val)
    train_size = len(split) - test_size - val_size

    if seed is not None:
        set_seed(seed)
    shuffle(split)
    train_set, val_set, test_set = split[:train_size], split[train_size:(train_size + test_size)], \
                                   split[(train_size + test_size):]
    assert len(set(train_set).intersection(set(test_set))) == 0
    assert len(set(val_set).intersection(set(test_set))) == 0
    assert len(set(train_set).intersection(set(test_set))) == 0
    return train_set, val_set, test_set
