"""
    This module provide the concept of Neural Network's Layer.
"""
import numpy as np


class _Layer:
    """
    Neural Network's layer implementation.

    Args:
        previous_nodes (int): number of previous nodes.
        nodes (int): number of layer's nodes.
        activation_function (function): layer's activation function.
        activation_function_derivative (function): derivative of activation_function argument.
    """
    def __init__(self, previous_nodes, nodes, activation_function, activation_function_derivative):
        self.__previous_nodes = previous_nodes
        self.__nodes = nodes
        self.__activation_function = activation_function
        self.__activation_function_derivative = activation_function_derivative
        self.__weights = np.random.rand(self.__nodes, self.__previous_nodes)
        self.__last_output = None
        self.__net = None

    def computes(self, inputs):
        """
        Computes the output of layer given the input.

        Args:
            inputs (list[int]): input vector given to the layer.

        Returns:
            matrix: result of computing the activation function of the layer's net.
        """
        # append 1 to input vector for bias and reshape as column vector
        inputs = inputs.reshape((len(inputs), 1))

        # performs the net of the layer
        self.__net = self.__weights.dot(
            inputs
        )

        # calculates the output of the layer
        self.__last_output = self.__activation_function(self.__net)

        return self.__last_output

    def get_previous_nodes(self):
        """
        Returns the number of previous layer's nodes.

        Returns:
            int: number of previous layer's nodes.
        """
        return self.__previous_nodes

    def get_nodes(self):
        """
        Returns the number of layer's nodes.

        Returns:
            int: number of layer's nodes.
        """
        return self.__nodes

    def get_activation_function_derivative(self):
        """
        Returns the layer's activation function derivative.

        Returns:
            function: activation function derivative.
        """
        return self.__activation_function_derivative

    def get_activation_function(self):
        """
        Returns the layer's activation function.

        Returns:
            function: activation function.
        """
        return self.__activation_function

    def get_weights(self):
        """
        Returns the weights of layer's nodes.

        Returns:
            matrix: weights of layer's nodes
        """
        return self.__weights

    def set_weights(self, weights):
        """
        Sets new weights of layer's nodes.

        Args:
            weights (matrix): new weights of layer's nodes.
        """
        self.__weights = weights

    def get_net(self):
        """
        Returns the net of the layer.

        Returns:
            matrix: net of the layer
        """
        return self.__net

    def get_last_output(self):
        """
        Returns last output produced by computes method.

        Returns:
             matrix: last output produced by computes method.
        """
        return self.__last_output
