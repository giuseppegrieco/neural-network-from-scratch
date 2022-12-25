from __future__ import annotations

from abc import abstractmethod, ABC

import numpy as np

import neural_network.learning_algorithm as learning_algorithm
from neural_network import NeuralNetwork


class LearningObserver(ABC):
    """
    The Learning Observer interface declares the update method, used by Learning Algorithms.
    """

    @abstractmethod
    def update(
            self,
            learning_algorithm: learning_algorithm.LearningAlgorithm,
            neural_network: NeuralNetwork,
            X_train: np.mat,
            Y_train: np.mat
    ) -> None:
        """
        This method is called at each iteration of the learning algorithm.

        :param learning_algorithm: LearningAlgorithm
        :param neural_network: NeuralNetwork
        :param X_train: np.mat
        :param Y_train: np.mat
        :return:
        """
        pass
