import numpy as np

from neural_network.functions.function import Function


class PolakRibiere(Function):
    @staticmethod
    def evaluate(x):
        gradient, last_gradient = x
        return (np.dot(gradient.T, gradient - last_gradient)) / np.power(np.linalg.norm(last_gradient, 2), 2)

    @staticmethod
    def derivative(x):
        return None
