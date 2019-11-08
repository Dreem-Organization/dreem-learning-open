import numpy as np

from ..datasets.data_augmentation import kill_channel

TEMPORAL_CONTEXT = 5
SIGNAL_LENGTH = 3000
N_CHANNELS = 5
TENSOR_SHAPE = (TEMPORAL_CONTEXT, N_CHANNELS, SIGNAL_LENGTH)


def test_kill_channel():
    X = np.random.random(size=TENSOR_SHAPE)
    assert np.linalg.norm(X - kill_channel(X, p=0)) == 0
    X = kill_channel(X, p=1)
    assert np.sum(np.sum(X, (0, 2)) == 0) >= 1
