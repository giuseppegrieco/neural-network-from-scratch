"""
    This module provide the concept of Artificial Neural Network.
"""
import numpy as np


class NeuralNetwork:
    """
    Artificial Neural Network implementation.

    Attributes:
        hyperparameters (:Hyperparameter): It contains all hyperparameters for the nn.
    """
    def __init__(self, hyperparameters):
        self.__hyperparameters = hyperparameters
        self.__init_weights()

    def __init_weights(self):
        """
        Computes the initial weights of the network.

        TODO: write the method
        """

    def train(self, input_data, expected_output):
        """
        Performs the training phase.

        TODO: write the method
        """
        output = self.feed_forward(input_data)
        self.__back_propagation(input_data, expected_output, output)

    def __back_propagation(self, input_data, target, output):
        """
        Performs back-propagation.

        TODO: fix the method (note output is already vector column)
        """
        # reshape input data as column vector
        input_data = np.mat(np.array(input_data, dtype=float)).T

        # reshape target vector as column vector
        target = np.mat(np.array(target)).T

        delta = -1 * (target - output)

        layers = self.__hyperparameters.get_hidden_layers().copy()
        layers.append(self.__hyperparameters.get_output_layer())

        for layer_index in range(len(layers) - 1, 0, -1):
            current_net = layers[layer_index].get_net()
            current_activation_function_derivative = layers[layer_index].get_activation_function_derivative()
            delta = delta * current_activation_function_derivative(
                current_net
            )
            delta_w = np.mat(delta).T * np.mat(layers[layer_index - 1].get_last_output())
            layers[layer_index].set_weights(
                layers[layer_index].get_weights() + (delta_w * self.__hyperparameters.get_learning_rate())
            )

        current_activation_function_derivative = layers[0].get_activation_function_derivative()
        delta = delta * current_activation_function_derivative(
            layers[0].get_net()
        )
        delta_w = np.mat(input_data).T * np.mat(delta)
        layers[0].set_weights(
            layers[0].get_weights() + (delta_w * self.__hyperparameters.get_learning_rate())
        )

    def feed_forward(self, nn_input):
        """
        Computes the new input and return the result.

        Args:
            nn_input (list[int]): Input gived to the neural network

        Returns:
            Neural Network output.
        """
        nn_input = np.array(nn_input, dtype=float)
        for layer in self.__hyperparameters.get_hidden_layers():
            nn_input = layer.computes(nn_input)

        return self.__hyperparameters.get_output_layer().computes(nn_input)
