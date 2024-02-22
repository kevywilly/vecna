import numpy as np


def normalize_mean_std(input: np.array, means: np.array, stds: np.array):
    return (input - means)/stds


def normalize_max(input: np.array, maxs: np.array):
    return input/np.max(maxs)
