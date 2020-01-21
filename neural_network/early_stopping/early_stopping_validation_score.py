import sys

import numpy as np

from neural_network import learning_algorithm as learning_algorithm, NeuralNetwork
from neural_network.early_stopping.early_stopping import EarlyStopping


class EarlyStoppingValidationScore(EarlyStopping):
    def __init__(
            self,
            max_fails: int
    ):
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
        if self._error_observer is not None and len(self._error_observer.store) > 0:
            new_error = self._error_observer.store[-1]
            if new_error < self.__minimum:
                self.__minimum = new_error
                self.__fail_counter = 0
            else:
                self.__fail_counter += 1
                if self.__fail_counter == self.__max_fails:
                    learning_algorithm.stopped = True
