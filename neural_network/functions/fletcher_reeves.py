import numpy as np

from neural_network.functions.function import Function


class FletcherReeves(Function):
    @staticmethod
    def evaluate(x):
        gradient, last_gradient = x
        return np.power(
            np.linalg.norm(gradient, 2) / np.linalg.norm(last_gradient, 2),
            2
        )

    @staticmethod
    def derivative(x):
        return None
