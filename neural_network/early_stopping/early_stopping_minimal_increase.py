import sys

import numpy as np

from neural_network import learning_algorithm as learning_algorithm, NeuralNetwork
from neural_network.early_stopping.early_stopping import EarlyStopping


class EarlyStoppingMinimalIncrease(EarlyStopping):
    def __init__(
            self,
            minimal_increase,
            max_fails
    ):
        super().__init__()
        self.__minimal_increase = minimal_increase
        self.__max_fails = max_fails
        self.__fail_counter = 0
        self.__last_error = sys.float_info.min

    def update(
            self,
            learning_algorithm,
            neural_network: NeuralNetwork,
            X_train: np.mat,
            Y_train: np.mat
    ) -> None:
        if self._error_observer is not None and len(self._error_observer.store) > 0:
            new_error = self._error_observer.store[-1]
            if self.__last_error - new_error < self.__last_error * self.__minimal_increase:
                self.__fail_counter += 1
            else:
                self.__fail_counter = 0
            if self.__fail_counter == self.__max_fails:
                learning_algorithm.stopped = True
            self.__last_error = new_error
