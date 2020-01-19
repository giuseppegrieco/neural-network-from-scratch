import sys

import numpy as np

import neural_network.learning_algorithm as learning_algorithm

from neural_network.functions import MeanSquaredError
from neural_network.layers import Layer, WeightsInitializer
from neural_network.neural_network_cc import NeuralNetworkCC


class CascadeCorrelation(learning_algorithm.LearningAlgorithm):
    def __init__(
            self,
            learning_rate: float,
            momentum: float,
            regularization_correlation: float,
            regularization_pseudo_inverse: float,
            activation_function,
            weights_initializer: WeightsInitializer,
            epochs: int,
            max_nodes: int,
            pool_size: int
    ):
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__regularization_correlation = regularization_correlation
        self.__regularization_pseudo_inverse = regularization_pseudo_inverse
        self.__activation_function = activation_function
        self.__weights_initializer = weights_initializer
        self.__epochs = epochs
        self.__last_output = None
        self.__max_nodes = max_nodes
        self.__pool_size = pool_size

    def train(self, neural_network: NeuralNetworkCC, X_train: np.mat, Y_train: np.mat):
        X_train = Layer.add_bias_input(X_train)
        number_of_nodes = 0
        current_input = X_train
        output_layer = neural_network.layers[-1]
        self.__update_output_layer(output_layer, Y_train, current_input)
        self.__last_output = output_layer.computes(current_input)
        while number_of_nodes < self.__max_nodes:
            correlation = sys.float_info.min
            hidden_layer = None
            for i in range(0, self.__pool_size):
                candidate, candidate_correlation = self.__candidate_unit(X_train, Y_train, current_input)
                if candidate_correlation > correlation:
                    correlation = candidate_correlation
                    hidden_layer = candidate

            current_input = np.vstack((
                current_input,
                hidden_layer.computes(current_input)
            ))
            self.__update_output_layer(output_layer, Y_train, current_input)

            neural_network.layers.pop(-1)
            neural_network.layers.append(hidden_layer)
            neural_network.layers.append(output_layer)
            number_of_nodes += 1

            self._notify()

    def __candidate_unit(self, X_train, Y_train, current_input):
        hidden_layer = Layer(
            1,
            self.__activation_function,
            self.__weights_initializer
        )
        self.__weights_initializer.initializes(
            hidden_layer,
            (1, len(current_input))
        )

        E = Y_train - self.__last_output
        E_mean = np.sum(E, axis=1) * 1 / len(X_train.T)
        E_mean = np.reshape(E_mean, (len(Y_train), 1))
        E_tot = E - E_mean

        new_correlation, V_tot = self.__calculates_correlation(
            hidden_layer.computes(current_input),
            E_tot
        )
        delta_old = np.zeros(hidden_layer.weights.shape)
        i = 10
        current_epoch = 0
        max_correlation = sys.float_info.min
        max_weights = hidden_layer.weights
        while i > 0 and current_epoch < self.__epochs:
            delta = np.sign(
                self.__calculates_output_correlation(V_tot, E_tot)
            )
            delta = np.reshape(delta, (len(Y_train), 1))
            delta = np.multiply(delta, E - E_mean)
            delta = np.multiply(delta, hidden_layer.activation_function.derivative(
                hidden_layer.net
            ))
            delta = np.sum(np.dot(delta, hidden_layer.last_input.T).T, axis=1)
            delta = np.reshape(delta, (1, len(current_input)))

            hidden_layer.weights = (
                    hidden_layer.weights +
                    (self.__learning_rate * delta) +
                    (self.__momentum * delta_old) +
                    (-self.__regularization_correlation * hidden_layer.weights)
            )
            delta_old = delta
            new_correlation, V_tot = self.__calculates_correlation(
                hidden_layer.computes(current_input),
                E_tot
            )
            if new_correlation > max_correlation:
                max_correlation = new_correlation
                max_weights = hidden_layer.weights
                i = 10
            else:
                i -= 1
            current_epoch += 1

        hidden_layer.weights = max_weights
        return hidden_layer, new_correlation

    @staticmethod
    def __calculates_output_correlation(V_tot, E_tot):
        return np.sum(
            np.multiply(V_tot, E_tot),
            axis=1
        )

    def __calculates_correlation(self, hidden_layer_output, E_tot):
        V = hidden_layer_output
        V_mean = np.mean(V)
        V_tot = V - V_mean

        return np.sum(
            np.absolute(self.__calculates_output_correlation(V_tot, E_tot)),
            axis=0
        ), V_tot

    def __update_output_layer(self, output_layer: Layer, Y_train: np.mat, new_input):
        pseudo_inverse = self.__pseudo_inverse(new_input).T
        new_weights = np.dot(
            Y_train,
            pseudo_inverse
        )
        self.__last_output = new_weights.dot(
            new_input
        )
        output_layer.weights = new_weights

    def __pseudo_inverse(self, Y_train):
        return np.dot(np.linalg.inv(
            np.dot(
                Y_train,
                Y_train.T
            ) + (np.identity(len(Y_train)) * -self.__regularization_pseudo_inverse)
        ), Y_train)