import numpy as np


def convert_in_numpy(target):
    if isinstance(target, list) and not isinstance(target[0], list):
        target = [1] + target
        target = np.array(target, dtype=float)
        return target.reshape((len(target), 1))
    else:
        target = np.array(target)
        n, m = target.shape
        bias = np.ones((1, m))
        return np.vstack((bias, target))
