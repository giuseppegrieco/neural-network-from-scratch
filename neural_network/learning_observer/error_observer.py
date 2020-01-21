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
        self._predicted_Y = neural_network.feed_forward(self._X)
        self._store.append(
            self._error_function.evaluate(self._Y - self._predicted_Y)
        )

    @property
    def store(self) -> List[float]:
        return self._store

    @store.setter
    def store(self, store: List[float]):
        self._store = store
