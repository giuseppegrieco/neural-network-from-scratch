import numpy as np

from neural_network.functions.function import Function


class MeanEuclideanError(Function):
    @staticmethod
    def evaluate(x):
        return np.mean(
            np.sqrt(
                np.sum(
                    np.power(x, 2),
                    axis=0
                )
            )
        )

    @staticmethod
    def derivative(x):
        return None
