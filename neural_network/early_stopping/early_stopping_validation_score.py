import sys

import numpy as np

from neural_network import learning_algorithm as learning_algorithm, NeuralNetwork
from neural_network.early_stopping.early_stopping import EarlyStopping


class EarlyStoppingValidationScore(EarlyStopping):
    def __init__(
            self,
            max_fails: int
    ):
        """
        This is the constructor for the class EarlyStoppingValidationScore.

        :param max_fails: int
        """
        super().__init__()
        self.__max_fails = max_fails
        self.__fail_counter = 0
        self.__minimum = sys.float_info.max

    def update(
            self,
            learning_algorithm,
            neural_network: NeuralNetwork,
            X_train: np.mat,
            Y_train: np.mat
    ) -> None:
        """
        This checks if the validations is growing, and if the condition is satisfy for more than max fails indicated
        the learning algorithm will be stopped.

        :param learning_algorithm: LearningAlgorithm
        :param neural_network: NeuralNetwork
        :param X_train: np.mat
        :param Y_train: np.mat
        :return:
        """
        if self._error_observer is not None and len(self._error_observer.store) > 0:
            new_error = self._error_observer.store[-1]
            if new_error < self.__minimum:
                self.__minimum = new_error
                self.__fail_counter = 0
            else:
                self.__fail_counter += 1
                if self.__fail_counter == self.__max_fails:
                    learning_algorithm.stopped = True
