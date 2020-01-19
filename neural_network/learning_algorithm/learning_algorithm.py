from __future__ import annotations

from abc import abstractmethod, ABC

import numpy as np

import neural_network as neural_network
import neural_network.learning_algorithm as learning_algorithm


class LearningAlgorithm(ABC):
    def __init__(self):
        self._learning_observers = []
        self._stopped = False

    @abstractmethod
    def train(self, neural_network: neural_network.NeuralNetwork, X_train: np.mat, Y_train: np.mat):
        pass

    def attach(self, learning_observer: learning_algorithm.LearningObserver) -> None:
        self._learning_observers.append(learning_observer)

    def detach(self, learning_observer: learning_algorithm.LearningObserver) -> None:
        self._learning_observers.remove(learning_observer)

    def detach_all(self) -> None:
        self._learning_observers.clear()

    def _notify(self):
        for learning_observer in self._learning_observers:
            learning_observer.update(self)

    @property
    def stopped(self) -> bool:
        return self._stopped

    @stopped.setter
    def stopped(self, stopped: bool):
        self._stopped = stopped

