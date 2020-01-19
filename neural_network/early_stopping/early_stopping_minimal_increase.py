import sys

from neural_network import learning_algorithm as learning_algorithm
from neural_network.early_stopping.early_stopping import EarlyStopping


class EarlyStoppingMinimalIncrease(EarlyStopping):
    def __init__(
            self,
            minimal_increase,
            max_fails
    ):
        super().__init__()
        self.__minimal_increase = minimal_increase
        self.__fail_counter = 0
        self.__last_error = sys.float_info.min
        self.__max_fails = max_fails

    def update(self, learning_algorithm: learning_algorithm.LearningAlgorithm) -> None:
        if len(self._error_observer.store) > 0:
            new_error = self._error_observer.store[-1]
            if self.__last_error - new_error < self.__last_error * self.__minimal_increase:
                self.__fail_counter += 1
            if self.__fail_counter == self.__max_fails:
                learning_algorithm.stopped = True
