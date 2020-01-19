from abc import ABC, abstractmethod

import numpy as np

from neural_network import NeuralNetwork
from neural_network.learning_algorithm import LearningAlgorithm


class CrossValidation(ABC):
    def __init__(self):
        self.on_finish = self._noop
        self.on_fold_attempt_start = self._noop
        self.on_fold_attempt_end = self._noop

    @abstractmethod
    def estimates(
            self,
            neural_network: NeuralNetwork,
            learning_algorithm: LearningAlgorithm,
            X_train: np.mat,
            Y_train: np.mat):
        pass

    def _noop(self, *params):
        pass

    @property
    def on_finish(self):
        return self._on_finish

    @on_finish.setter
    def on_finish(self, on_finish):
        self._on_finish = on_finish

    @property
    def on_fold_attempt_start(self):
        return self._on_finish

    @on_fold_attempt_start.setter
    def on_fold_attempt_start(self, on_fold_attempt_start):
        self._on_fold_attempt_start = on_fold_attempt_start

    @property
    def on_fold_attempt_end(self):
        return self._on_fold_attempt_end

    @on_fold_attempt_end.setter
    def on_fold_attempt_end(self, on_fold_attempt_end):
        self._on_fold_attempt_end = on_fold_attempt_end
