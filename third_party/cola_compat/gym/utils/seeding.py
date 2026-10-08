import numpy as np


def np_random(seed=None):
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31))
    return np.random.RandomState(seed), seed
