import numpy as np

from neural_network.functions.function import Function


class TanH(Function):
    @staticmethod
    def evaluate(x):
        return np.tan(x)

    @staticmethod
    def derivative(x):
        return 1 - np.square(TanH.evaluate(x))
