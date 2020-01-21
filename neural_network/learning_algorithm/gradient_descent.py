from typing import List

import numpy as np

import neural_network as neural_network
import neural_network.learning_algorithm as learning_algorithm


class GradientDescent(learning_algorithm.LearningAlgorithm):
    def __init__(
            self,
            learning_rate: float,
            momentum: float,
            regularization: float,
            epochs: int
    ):
        super().__init__()
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._regularization = regularization
        self._epochs = epochs

    def train(self, neural_network: neural_network.NeuralNetwork, X_train: np.mat, Y_train: np.mat):
        super().train(neural_network, X_train, Y_train)
        self._stopped = False
        momentum_memory = [0] * len(neural_network.layers)
        i = 000
        while i < self._epochs and not self._stopped:
            predicted_Y = neural_network.feed_forward(X_train)

            gradients = self.__back_propagation(neural_network, Y_train, predicted_Y)
            j = 0
            for gradient in reversed(gradients):
                momentum_memory[j] = self.update_weights(
                    neural_network.layers[j],
                    gradient,
                    momentum_memory[j],
                    len(X_train.T)
                )
                j += 1

            self._notify(neural_network, X_train, Y_train)
            i += 1

    @staticmethod
    def __back_propagation(neural_network: neural_network.NeuralNetwork, Y_train, predicted_Y) -> List[np.mat]:
        gradients = []

        delta = - (Y_train - predicted_Y)
        for layer in reversed(neural_network.layers):
            delta = np.multiply(delta, layer.activation_function.derivative(
                layer.net
            ))
            gradients.append(
                np.dot(delta, layer.last_input.T)
            )
            delta = np.dot(delta.T, layer.weights)
            delta = delta[:, 1:].T

        return gradients

    def update_weights(self, layer, gradient: np.mat, momentum_stored, n_pattern):
        layer_weights = layer.weights

        lambda_mat = np.full(layer_weights.shape, -self._regularization)
        if layer.is_hidden:
            lambda_mat[:, 0] = 0.0
        regularization = np.multiply(lambda_mat, layer_weights)

        delta_w = (-self._learning_rate * 1 / n_pattern * gradient) + (self._momentum * momentum_stored) + regularization

        layer.weights = layer_weights + delta_w

        return delta_w
