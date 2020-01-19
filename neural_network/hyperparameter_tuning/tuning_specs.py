from abc import ABC, abstractmethod

from neural_network import NeuralNetwork
from neural_network.learning_algorithm import LearningAlgorithm


class TuningSpecs(ABC):
    @abstractmethod
    def build_neural_network_object(self, hyperparameters: tuple) -> NeuralNetwork:
        pass

    @abstractmethod
    def build_learning_algorithm_object(self, hyperparameters: tuple) -> LearningAlgorithm:
        pass

    @abstractmethod
    def combinations_of_hyperparameters(self):
        pass
