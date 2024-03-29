from typing import List

from neural_network.hyperparameter_tuning import TuningSpecs
from neural_network.layers import Layer, WeightsInitializer
from neural_network.learning_algorithm import CascadeCorrelation
from neural_network.neural_network_cc import NeuralNetworkCC


class CascadeCorrelationTuningSpecs(TuningSpecs):
    def __init__(
            self,
            input_size: int,
            output_layer_list: List[Layer],
            learning_rate_list: List[float],
            momentum_list: List[float],
            regularization_correlation_list: List[float],
            regularization_pseudo_inverse_list: List[float],
            max_nodes_list: List[int],
            pool_size_list: List[int],
            epochs_list: List[int],
            weights_initializer_list: List[WeightsInitializer],
            activation_function_list: List,
            minimal_correlation_increase_list: List[float],
            max_fails_increase_list: List[int]
    ):
        """
        This is the constructor for the class CascadeCorrelationTuningSpecs.

        :param input_size: int
        :param output_layer_list: List[Layer]
        :param learning_rate_list: List[float]
        :param momentum_list: List[float]
        :param regularization_correlation_list: List[float]
        :param regularization_pseudo_inverse_list: List[float]
        :param max_nodes_list: List[int]
        :param pool_size_list: List[int]
        :param epochs_list: List[int]
        :param weights_initializer_list: List[WeightsInitializer]
        :param activation_function_list: List
        :param minimal_correlation_increase_list: List[float]
        :param max_fails_increase_list: List[int]
        """
        self.__input_size = input_size
        self.__output_layer_list = output_layer_list
        self.__learning_rate_list = learning_rate_list
        self.__momentum_list = momentum_list
        self.__regularization_correlation_list = regularization_correlation_list
        self.__regularization_pseudo_inverse_list = regularization_pseudo_inverse_list
        self.__max_nodes_list = max_nodes_list
        self.__pool_size_list = pool_size_list
        self.__epochs_list = epochs_list
        self.__weights_initializer_list = weights_initializer_list
        self.__activation_function_list = activation_function_list
        self.__minimal_correlation_increase_list = minimal_correlation_increase_list
        self.__max_fails_increase_list = max_fails_increase_list

    def build_neural_network_object(self, hyperparameters: List) -> NeuralNetworkCC:
        """
        Builds the NeuralNetworkCC object given a hyperparameters list.

        :param hyperparameters: List

        :return: NeuralNetworkCC
        """
        return NeuralNetworkCC(
            self.__input_size,
            [hyperparameters[9]]
        )

    def build_learning_algorithm_object(self, hyperparameters: List) -> CascadeCorrelation:
        """
        Builds the CascadeCorrelation object given a hyperparameter list.

        :param hyperparameters: List

        :return: CascadeCorrelation
        """
        return CascadeCorrelation(
            activation_function=hyperparameters[0],
            weights_initializer=hyperparameters[1],
            epochs=hyperparameters[2],
            pool_size=hyperparameters[3],
            max_nodes=hyperparameters[4],
            learning_rate=hyperparameters[5],
            momentum=hyperparameters[6],
            regularization_correlation=hyperparameters[7],
            regularization_pseudo_inverse=hyperparameters[8],
            minimal_correlation_increase=hyperparameters[10],
            max_fails_increase=hyperparameters[11]
        )

    def combinations_of_hyperparameters(self) -> List:
        return [
            [
                activation_function, weights_initializer, epochs, pool_size, max_nodes, learning_rate,
                momentum, regularization_correlation, regularization_pseudo_inverse, output_layer,
                minimal_correlation_increase, max_fails_increase
            ]
            for output_layer in self.__output_layer_list
            for activation_function in self.__activation_function_list
            for weights_initializer in self.__weights_initializer_list
            for epochs in self.__epochs_list
            for pool_size in self.__pool_size_list
            for max_nodes in self.__max_nodes_list
            for learning_rate in self.__learning_rate_list
            for momentum in self.__momentum_list
            for regularization_correlation in self.__regularization_correlation_list
            for regularization_pseudo_inverse in self.__regularization_pseudo_inverse_list
            for minimal_correlation_increase in self.__minimal_correlation_increase_list
            for max_fails_increase in self.__max_fails_increase_list
        ]

    def combinations_repr(self, hyperparameters: List):
        return {
            "activation_function": hyperparameters[0].__name__,
            "weights_initializer": type(hyperparameters[1]).__name__,
            "epochs": hyperparameters[2],
            "pool_size": hyperparameters[3],
            "max_nodes": hyperparameters[4],
            "learning_rate": hyperparameters[5],
            "momentum": hyperparameters[6],
            "regularization_correlation": hyperparameters[7],
            "regularization_pseudo_inverse": hyperparameters[8],
            "output_layer": hyperparameters[9].__repr__(),
            'minimal_correlation_increase': hyperparameters[10],
            'max_fails_increase': hyperparameters[11]
        }
