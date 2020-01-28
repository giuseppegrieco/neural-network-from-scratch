from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy

import numpy as np

from neural_network import NeuralNetwork
from neural_network.early_stopping.early_stopping import EarlyStopping
from neural_network.learning_algorithm import LearningAlgorithm
from neural_network.learning_observer import ErrorObserver


class CrossValidation(ABC):
    def __init__(self):
        self._original_early_stopping_list = []
        self._early_stopping_list = []

    @abstractmethod
    def estimates(
            self,
            neural_network: NeuralNetwork,
            learning_algorithm: LearningAlgorithm
    ):
        """
        This method will implements cross validation routine.

        :param neural_network: NeuralNetwork
        :param learning_algorithm: LearningAlgorithm
        """
        pass

    def _attach_early_stopping(self, validation_observer: ErrorObserver, learning_algorithm: LearningAlgorithm):
        """
        This method allows to attach early stopping list to a learning algorithm.

        :param validation_observer: ErrorObserver
        :param learning_algorithm: Learning Algorithm
        """
        self._early_stopping_list.clear()
        for learning_observer in self._original_early_stopping_list:
            self._early_stopping_list.append(copy(learning_observer))
        for early_stopping in self._early_stopping_list:
            early_stopping.error_observer = validation_observer
            learning_algorithm.attach(early_stopping)

    def _detach_early_stopping(self, learning_algorithm: LearningAlgorithm):
        """
        Thismethod allows to detach a early stopping list from a learning algorithm.

        :param learning_algorithm: LearningAlgorithm
        """
        for early_stopping in self._early_stopping_list:
            learning_algorithm.detach(early_stopping)

    def add_early_stopping(self, early_stopping: EarlyStopping) -> None:
        """
        This method allows to attach early stopping.
        """
        self._original_early_stopping_list.append(early_stopping)

    def remove_early_stopping(self, early_stopping: EarlyStopping) -> None:
        """
        This method allows to detach early stopping.
        """
        self._original_early_stopping_list.remove(early_stopping)
