from typing import List

import numpy as np

from neural_network import NeuralNetwork
from neural_network.learning_observer import LearningObserver


class ErrorObserver(LearningObserver):
    def __init__(self, X: np.mat, Y: np.mat, error_function):
        self.store = []
        self._X = X
        self._Y = Y
        self._error_function = error_function
        self._predicted_Y = None

    def update(
            self,
            learning_algorithm,
            neural_network: NeuralNetwork,
            X_train: np.mat,
            Y_train: np.mat
    ) -> None:
        """
        This method is called at each iteration of the learning algorithm and calculates the error.

        :param learning_algorithm: LearningAlgorithm
        :param neural_network: NeuralNetwork
        :param X_train: np.mat
        :param Y_train: np.mat
        """
        self._predicted_Y = neural_network.feed_forward(self._X)
        self._store.append(
            self._error_function.evaluate(self._Y - self._predicted_Y)
        )

    @property
    def store(self) -> List[float]:
        """
        Returns the list of error

        :returns: List[float]
        """
        return self._store

    @store.setter
    def store(self, store: List[float]):
        """
        Allows to set a list of error.

        :param store: List[float]
        """
        self._store = store
