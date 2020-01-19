import numpy as np

from neural_network.functions.function import Function


class HestenesStiefel(Function):
    @staticmethod
    def evaluate(x):
        gradient, last_gradient, last_direction = x
        return np.dot(gradient.T, gradient - last_gradient) / np.dot((gradient - last_gradient).T, last_direction)

    @staticmethod
    def derivative(x):
        return None
