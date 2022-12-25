import numpy as np

from neural_network.functions.function import Function


class MeanSquaredError(Function):
    @staticmethod
    def evaluate(x):
        return np.mean(
            np.power(x, 2)
        )

    @staticmethod
    def derivative(x):
        return - x
