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
        """
        This method will specify behaviors of learning algorithm.

        :param neural_network: NeuralNetwork
        :param X_train: np.mat
        :param Y_train: np.mat
        """
        self._learning_observers.clear()
        for learning_observer in self._original_learning_observers:
            self._learning_observers.append(copy(learning_observer))

    def attach(self, learning_observer: learning_observer.LearningObserver) -> None:
        """
        This method allows to attach a Learning Observer.

        :param learning_observer: LearningObserver
        """
        self._original_learning_observers.append(learning_observer)

    def detach(self, learning_observer: learning_observer.LearningObserver) -> None:
        """
        This method allows to remove a learning observer from the list.

        :param learning_observer: LearningObserver
        """
        self._original_learning_observers.remove(learning_observer)

    def detach_all(self) -> None:
        """
        This method allows to clear the list of learning observer.

        :param learning_observer: LearningObserver
        """
        self._original_learning_observers.clear()

    def _notify(self, neural_network: neural_network.NeuralNetwork, X_train: np.mat, Y_train: np.mat):
        """
        This method allows to notify all learning observer contained in the list.
        """
        for learning_observer in self._learning_observers:
            learning_observer.update(self, neural_network, X_train, Y_train)

    @property
    def stopped(self) -> bool:
        """
        This indicates if the learning algorithm is stopped or not.

        :return: bool
        """
        return self._stopped

    @stopped.setter
    def stopped(self, stopped: bool):
        """
        This allows to stop a learning algorithm.

        :param stopped: bool
        """
        self._stopped = stopped

