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
        # reshape input vector as column vector
        input_data = np.array(input_data, dtype=float).reshape((len(input_data), 1))

        # reshape target vector as column vector
        target = np.mat(np.array(target)).T

        delta = -1 * (target - output)

        output_layer = self.__hyperparameters.get_output_layer()
        hidden_layer = self.__hyperparameters.get_hidden_layers()[0]

        delta = np.multiply(delta, output_layer.get_activation_function_derivative()(
            output_layer.get_net()
        ))
        delta_oh = delta * hidden_layer.get_last_output().T
        output_layer.set_weights(output_layer.get_weights() + (delta_oh * -self.__hyperparameters.get_learning_rate()))

        delta = delta.T * output_layer.get_weights()
        delta = np.multiply(delta.T, hidden_layer.get_activation_function_derivative()(
            hidden_layer.get_net()
        ))
        delta_hi = delta * input_data.T
        hidden_layer.set_weights(hidden_layer.get_weights() + (delta_hi * -self.__hyperparameters.get_learning_rate()))

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
