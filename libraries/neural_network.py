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

    def feedforward(self, nn_input):
        """
        Computes the new input and return the result.

        Args:
            nn_input (list[int]): Input gived to the neural network

        Returns:
            Neural Network output.
        """
        nn_input = np.array(nn_input)
        for layer in self.__hyperparameters.get_hidden_layers():
            nn_input = layer.computes(nn_input)

        return self.__hyperparameters.get_output_layer().computes(nn_input)
