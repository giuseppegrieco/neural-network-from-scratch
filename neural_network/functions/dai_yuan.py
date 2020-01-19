import numpy as np

from neural_network.functions.function import Function


class DaiYuan(Function):
    @staticmethod
    def evaluate(x):
        gradient, last_gradient, last_direction = x
        return np.power(np.linalg.norm(gradient), 2) / np.dot((gradient - last_gradient).T, last_direction)

    @staticmethod
    def derivative(x):
        return None
