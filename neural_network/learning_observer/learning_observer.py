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
        pass
