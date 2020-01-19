from typing import List

from neural_network import NeuralNetwork
from neural_network.hyperparameter_tuning import TuningSpecs
from neural_network.layers import Layer
from neural_network.learning_algorithm import GradientDescent


class GradientDescentTuningSpecs(TuningSpecs):
    def __init__(
            self,
            input_size: int,
            layers_list: List[List[Layer]],
            learning_rate_list: List[float],
            momentum_list: List[float],
            regularization_list: List[float],
            epochs_list: List[int]
    ):
        self.__input_size = input_size
        self.__layers_list = layers_list
        self.__learning_rate_list = learning_rate_list
        self.__momentum_list = momentum_list
        self.__regularization_list = regularization_list
        self.__epochs_list = epochs_list

    def build_neural_network_object(self, hyperparameters) -> NeuralNetwork:
        return NeuralNetwork(
            self.__input_size,
            hyperparameters[1]
        )

    def build_learning_algorithm_object(self, hyperparameters: List) -> GradientDescent:
        return GradientDescent(
            epochs=hyperparameters[0],
            learning_rate=hyperparameters[2],
            momentum=hyperparameters[3],
            regularization=hyperparameters[4]
        )

    def combinations_of_hyperparameters(self) -> List:
        return [
            [epochs, layers, learning_rate, momentum, regularization]
            for epochs in self.__epochs_list
            for layers in self.__layers_list
            for learning_rate in self.__learning_rate_list
            for momentum in self.__momentum_list
            for regularization in self.__regularization_list
        ]

    def combinations_repr(self, hyperparameters: List):
        return {
            "epochs": hyperparameters[0],
            "layers": [layer.__repr__() for layer in hyperparameters[1]],
            "learning_rate": hyperparameters[2],
            "momentum": hyperparameters[3],
            "regularization": hyperparameters[4]
        }
