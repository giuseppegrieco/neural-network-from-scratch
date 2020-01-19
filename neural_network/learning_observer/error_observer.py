from typing import List

import numpy as np

from neural_network import NeuralNetwork
from neural_network.learning_algorithm import LearningAlgorithm
from neural_network.learning_observer import LearningObserver


class ErrorObserver(LearningObserver):
    def __init__(self, neural_network: NeuralNetwork, X: np.mat, Y: np.mat, error_function):
        self.store = []
        self.__neural_network = neural_network
        self._X = X
        self._Y = Y
        self.__error_function = error_function

    def update(self, learning_algorithm: LearningAlgorithm) -> None:
        predicted_Y = self.__neural_network.feed_forward(self._X)
        self._store.append(
            self.__error_function.evaluate(self._Y - predicted_Y)
        )

    @property
    def store(self) -> List[float]:
        return self._store

    @store.setter
    def store(self, store: List[float]):
        self._store = store
