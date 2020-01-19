import sys

from neural_network import learning_algorithm as learning_algorithm
from neural_network.learning_observer import ErrorObserver
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

    def update(self, learning_algorithm: learning_algorithm.LearningAlgorithm) -> None:
        new_error = self._error_observer.store[-1]
        if new_error < self.__minimum:
            self.__minimum = new_error
            self.__fail_counter = 0
        else:
            self.__fail_counter += 1
            if self.__fail_counter == self.__max_fails:
                learning_algorithm.stopped = True
