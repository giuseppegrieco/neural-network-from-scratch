from abc import ABC, abstractmethod
from typing import List

from neural_network import NeuralNetwork
from neural_network.learning_algorithm import LearningAlgorithm


class TuningSpecs(ABC):
    @abstractmethod
    def build_neural_network_object(self, hyperparameters: List) -> NeuralNetwork:
        """
        Builds the NeuralNetwork object given a hyperparameters list.

        :param hyperparameters: List

        :return: NeuralNetwork
        """
        pass

    @abstractmethod
    def build_learning_algorithm_object(self, hyperparameters: List) -> LearningAlgorithm:
        """
        Builds the LearningAlgorithm object given a hyperparameter list.

        :param hyperparameters: List

        :return: LearningAlgorithm
        """
        pass

    @abstractmethod
    def combinations_of_hyperparameters(self) -> List:
        """
        Returns all the combination of hyperparameters.

        :return: List
        """
        pass

    @abstractmethod
    def combinations_repr(self, hyperparameters: List):
        pass
