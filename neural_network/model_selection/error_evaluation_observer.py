from typing import List

import numpy as np

from neural_network import NeuralNetwork
from neural_network.learning_observer import LearningObserver, ErrorObserver


class ErrorEvaluationObserver(ErrorObserver):
    def __init__(self, X: np.mat, Y: np.mat, error_curve_function, error_evaluate_function):
        self.store = []
        self._X = X
        self._Y = Y
        self._error_function = error_curve_function
        self.__error_evaluate_function = error_evaluate_function
        self.store_evaluate = []

    def update(
            self,
            learning_algorithm,
            neural_network: NeuralNetwork,
            X_train: np.mat,
            Y_train: np.mat
    ) -> None:
        super().update(learning_algorithm, neural_network, X_train, Y_train)
        self.__store_evaluate.append(
            self.__error_evaluate_function.evaluate(self._Y - self._predicted_Y)
        )

    @property
    def store_evaluate(self) -> List[float]:
        return self.__store_evaluate

    @store_evaluate.setter
    def store_evaluate(self, store: List[float]):
        self.__store_evaluate = store

