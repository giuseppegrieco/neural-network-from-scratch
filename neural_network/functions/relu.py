import numpy as np

from neural_network.functions.function import Function


class ReLU(Function):
    @staticmethod
    def evaluate(x):
        return np.maximum(x, 0, x)

    @staticmethod
    def derivative(x):
        x[x >= 0] = 1
        x[x < 0] = 0
        return x
