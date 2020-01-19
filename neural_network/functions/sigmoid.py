import numpy as np

from neural_network.functions.function import Function


class Sigmoid(Function):
    @staticmethod
    def evaluate(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        y = Sigmoid.evaluate(x)
        return np.multiply(y, (1 - y))
