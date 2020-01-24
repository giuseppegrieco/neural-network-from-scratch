import logging
import sys

import numpy as np

import neural_network.learning_algorithm as learning_algorithm

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
            pool_size: int,
            minimal_correlation_increase: float,
            max_fails_increase: int
    ):
        super().__init__()
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__regularization_correlation = regularization_correlation
        self.__regularization_pseudo_inverse = regularization_pseudo_inverse
        self.__activation_function = activation_function
        self.__weights_initializer = weights_initializer
        self.__epochs = epochs
        self.__max_nodes = max_nodes
        self.__pool_size = pool_size
        self.__minimal_correlation_increase = minimal_correlation_increase
        self.__max_fails_increase = max_fails_increase

    def train(self, neural_network: NeuralNetworkCC, X_train: np.mat, Y_train: np.mat):
        super().train(neural_network, X_train, Y_train)
        self._stopped = False
        # Initializations
        X_train = Layer.add_bias_input(X_train)
        number_of_nodes = 1
        current_input = X_train
        output_layer = neural_network.layers[-1]
        last_output = self.__update_output_layer(output_layer, Y_train, current_input)

        # Iterate until reach max number of nodes or algorithm is stopped from outside
        while number_of_nodes <= self.__max_nodes and not self._stopped:
            logging.debug('Node n.%d' % number_of_nodes)

            logging.debug('Start pooling of %d candidate units' % self.__pool_size)
            # Find best unit candidates (based on max correlation) from a pool
            hidden_layer = None
            correlation = sys.float_info.min
            for i in range(1, self.__pool_size + 1):
                candidate, candidate_correlation = self.__candidate_unit(X_train, Y_train, current_input, last_output)
                logging.debug('Candidate n.%d, correlation=%f' % (i, candidate_correlation))
                if candidate_correlation > correlation:
                    correlation = candidate_correlation
                    hidden_layer = candidate

            logging.debug('Chosen candidate correlation=%f' % correlation)

            # Calculates new input of output layer (previous with the new unit)
            current_input = np.vstack((
                current_input,
                hidden_layer.computes(current_input)
            ))

            # Update the weights of output layer with pseudo-inverse
            last_output = self.__update_output_layer(output_layer, Y_train, current_input)

            # Add new candidate unit inside the network
            neural_network.layers.pop(-1)
            neural_network.layers.append(hidden_layer)
            neural_network.layers.append(output_layer)

            number_of_nodes += 1
            self._notify(neural_network, X_train, Y_train)

    def __candidate_unit(self, X_train, Y_train, current_input, last_output):
        # Creates the candidate units layer
        hidden_layer = Layer(
            1,
            self.__activation_function,
            self.__weights_initializer
        )
        self.__weights_initializer.initializes(
            hidden_layer,
            (1, len(current_input))
        )

        # Calculates
        E = Y_train - last_output
        E_mean = np.sum(E, axis=1) * 1 / len(X_train.T)
        E_mean = np.reshape(E_mean, (len(Y_train), 1))
        E_tot = E - E_mean

        # Calculates initial correlation
        new_correlation, V_tot = self.__calculates_correlation(
            hidden_layer.computes(current_input),
            E_tot
        )

        # Initializes momentum
        momentum_store = np.zeros(hidden_layer.weights.shape)

        # Initializations to limits of iterations
        i = self.__max_fails_increase
        current_epoch = 1
        max_correlation = sys.float_info.min
        max_weights = hidden_layer.weights

        logging.debug('Maximizes the correlation of the candidate node')
        # Iterates to try to increase the correlation
        while i > 0 and current_epoch <= self.__epochs:
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

            # Update the weights
            delta_w = (self.__learning_rate * 1 / len(X_train.T) * delta) + \
                      (self.__momentum * momentum_store)
            hidden_layer.weights = (
                    hidden_layer.weights +
                    delta_w + (-self.__regularization_correlation * hidden_layer.weights)
            )
            momentum_store = delta_w

            # Computes the new correlation
            new_correlation, V_tot = self.__calculates_correlation(
                hidden_layer.computes(current_input),
                E_tot
            )
            logging.debug('{ epochs: %d, correlation: %f }' % (current_epoch, new_correlation))

            # Checks that the correlation has increased enough
            if new_correlation - max_correlation < max_correlation * self.__minimal_correlation_increase:
                i -= 1
            else:
                i = self.__max_fails_increase

            # Save the max correlation and its weights
            if new_correlation > max_correlation:
                max_correlation = new_correlation
                max_weights = hidden_layer.weights
            current_epoch += 1

        # Update the weights of candidate with the weights of step that has max correlation
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
        last_output = new_weights.dot(
            new_input
        )
        output_layer.weights = new_weights
        return last_output

    def __pseudo_inverse(self, Y_train):
        return np.dot(np.linalg.inv(
            np.dot(
                Y_train,
                Y_train.T
            ) + (np.identity(len(Y_train)) * -self.__regularization_pseudo_inverse)
        ), Y_train)
