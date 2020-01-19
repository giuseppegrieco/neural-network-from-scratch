import numpy as np

from neural_network.learning_algorithm import GradientDescent


class ConjugateGradientDescent(GradientDescent):
    def __init__(
            self,
            learning_rate: float,
            regularization: float,
            epochs: int,
            beta_formula
    ):
        super().__init__()
        super().__init__(learning_rate, 0, regularization, epochs)
        self.__beta_formula = beta_formula


    def update_weights(self, layer, gradient: np.mat, momentum_stored):
        self._momentum = formula
        super().update_weights(layer, gradient, momentum_stored)
