import numpy as np


class NeuralNetwork:
    """
    Artificial Neural Network implementation.

    Attributes:
        hyperparameters (:Hyperparameter): It contains all hyperparameters for the nn.
    """
    def __init__(self, hyperparameters):
        """
        Inits NeuralNetwork with the hyperparameters indicated.

        Args:
            hyperparameters (Hyperparameter): It contains all hyperparameters to use for this nn.
        """
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
        ff = self.feedforward(input_data)
        self.backpropagation(input_data, expected_output, ff)

    def backpropagation(self, input_data, target, output):
        input_data = np.array(input_data, dtype=float)
        target = np.array(target)
        delta = -1 * (target - output)

        layers = self.__hyperparameters.get_hidden_layers().copy()
        layers.append(self.__hyperparameters.get_output_layer())

        for l in range(len(layers) - 1, 0, -1):
            current_net = layers[l].get_net()
            current_activation_function_derivative = layers[l].get_activation_function_derivative()

            delta = delta * current_activation_function_derivative(
                current_net
            )

            print(delta)
            delta_w = delta * (layers[l-1].get_last_output().transpose())

        current_activation_function_derivative = layers[0].get_activation_function_derivative()
        delta_w = delta * current_activation_function_derivative(
            layers[0].get_net()
        )
        delta_w = delta_w * input_data

    def feedforward(self, nn_input):
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
