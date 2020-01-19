import numpy as np

from neural_network.functions.function import Function


class Identity(Function):
    @staticmethod
    def evaluate(x):
        return x

    @staticmethod
    def derivative(x):
        return np.ones(
            x.shape
        )
