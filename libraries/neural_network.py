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
        #input_data = np.append(np.array([1]), input_data)
        output = self.feed_forward(input_data)
        input_data = [1] + input_data
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

        output_layer = self.__hyperparameters.get_output_layer()
        hidden_layers = self.__hyperparameters.get_hidden_layers()
        first_hidden_layer = hidden_layers[-1]

        # error of each output nodes

        output = output.reshape((len(output), 1))
        delta = -1 * (target - output)

        delta = np.multiply(delta, output_layer.get_activation_function_derivative()(
            output_layer.get_net()
        ))

        #adjusting delta of weights between last hidden layer and the output layer
        delta_oh = delta * first_hidden_layer.get_last_output().T
        output_layer.set_weights(output_layer.get_weights() + (delta_oh * -self.__hyperparameters.get_learning_rate()))

        previous_weights = output_layer.get_weights()
        for hidden_layer_index in range(len(hidden_layers) - 1, 0, -1):
            # TODO: forse meglio fare output_layer.get_weights().T * delta così da non trasporre due volte delta
            delta = delta.T * previous_weights
            delta = np.delete(delta, [0])
            delta = np.multiply(delta.T,hidden_layers[hidden_layer_index].get_activation_function_derivative()(
                hidden_layers[hidden_layer_index].get_net()
            ))
            deltah_h = delta * hidden_layers[hidden_layer_index - 1].get_last_output().T
            hidden_layers[hidden_layer_index].set_weights(
                hidden_layers[hidden_layer_index].get_weights() +
                deltah_h * - self.__hyperparameters.get_learning_rate()
            )
            previous_weights = hidden_layers[hidden_layer_index].get_weights()

        #TODO: forse meglio fare output_layer.get_weights().T * delta così da non trasporre due volte delta
        delta = delta.T * previous_weights
        delta = np.delete(delta, [0])
        delta = np.multiply(delta.T, hidden_layers[0].get_activation_function_derivative()(
            hidden_layers[0].get_net()
        ))

        # adjusting delta of weights between last hidden layer and the output layer
        delta_hi = delta * input_data.T
        hidden_layers[0].set_weights(hidden_layers[0].get_weights() + (delta_hi * -self.__hyperparameters.get_learning_rate()))

    def feed_forward(self, nn_input):
        """
        Computes the new input and return the result.

        Args:
            nn_input (list[int]): Input gived to the neural network

        Returns:
            Neural Network output.
        """
        nn_input = [1] + nn_input
        nn_input = np.array(nn_input, dtype=float)
        for layer in self.__hyperparameters.get_hidden_layers():
            nn_input = layer.computes(nn_input)

        return self.__hyperparameters.get_output_layer().computes(nn_input)
