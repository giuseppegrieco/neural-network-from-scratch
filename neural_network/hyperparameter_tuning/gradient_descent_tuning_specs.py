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
            epochs: int
    ):
        self.__input_size = input_size
        self.__layers_list = layers_list
        self.__learning_rate_list = learning_rate_list
        self.__momentum_list = momentum_list
        self.__regularization_list = regularization_list
        self.__epochs = epochs

    def build_neural_network_object(self, hyperparameters) -> NeuralNetwork:
        return NeuralNetwork(
            self.__input_size,
            hyperparameters[0]
        )

    def build_learning_algorithm_object(self, hyperparameters) -> GradientDescent:
        return GradientDescent(
            learning_rate=hyperparameters[1],
            momentum=hyperparameters[2],
            regularization=hyperparameters[3],
            epochs=self.__epochs
        )

    def combinations_of_hyperparameters(self):
        return [
            [layers, learning_rate, momentum, regularization]
            for layers in self.__layers_list
            for learning_rate in self.__learning_rate_list
            for momentum in self.__momentum_list
            for regularization in self.__regularization_list
        ]
