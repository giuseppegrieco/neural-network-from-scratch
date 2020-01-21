from __future__ import annotations
from copy import copy

from abc import abstractmethod, ABC

import numpy as np

import neural_network as neural_network
import neural_network.learning_observer as learning_observer


class LearningAlgorithm(ABC):
    def __init__(self):
        self._learning_observers = []
        self._original_learning_observers = []
        self._stopped = False

    @abstractmethod
    def train(self, neural_network: neural_network.NeuralNetwork, X_train: np.mat, Y_train: np.mat):
        self._learning_observers.clear()
        for learning_observer in self._original_learning_observers:
            self._learning_observers.append(copy(learning_observer))

    def attach(self, learning_observer: learning_observer.LearningObserver) -> None:
        self._original_learning_observers.append(learning_observer)

    def detach(self, learning_observer: learning_observer.LearningObserver) -> None:
        self._original_learning_observers.remove(learning_observer)

    def detach_all(self) -> None:
        self._original_learning_observers.clear()

    def _notify(self, neural_network: neural_network.NeuralNetwork, X_train: np.mat, Y_train: np.mat):
        for learning_observer in self._learning_observers:
            learning_observer.update(self, neural_network, X_train, Y_train)

    @property
    def stopped(self) -> bool:
        return self._stopped

    @stopped.setter
    def stopped(self, stopped: bool):
        self._stopped = stopped

