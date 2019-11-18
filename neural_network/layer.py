"""
    This module provide the concept of Neural Network's Layer.
"""
import numpy as np


class Layer:
    def __init__(self, nodes, activation_function):
        # Checks the correctness of parameter: nodes
        try:
            int(nodes)
            if nodes <= 0:
                raise ValueError('nodes parameter must be greater than 0')
        except ValueError:
            raise ValueError('nodes parameter must be an integer')
        self.__nodes = nodes

        # TODO: Check activation function correctness
        self.__activation_function = activation_function

        # Initialization
        self.__weights = None
        self.__net = None
        self.__last_output = None
        self.__is_hidden = None
        self.__delta_weights = None

    def computes(self, input_data):
        input_data = input_data.reshape((len(input_data), 1))

        # performs the net of the layer
        self.__net = self.__weights.dot(
            input_data
        )

        # calculates the output of the layer
        self.__last_output = self.__activation_function.f(self.__net)
        if self.__is_hidden:
            self.__last_output = np.append(np.array([1]), self.__last_output)
        self.__last_output = self.__last_output.reshape(len(self.__last_output), 1)
        return self.__last_output

    def set_weights(self, weights):
        self.__weights = weights

    def set_delta_weights(self, delta_weights):
        self.__delta_weights = delta_weights

    def set_is_hidden(self, is_hidden):
        self.__is_hidden = is_hidden

    def get_delta_weights(self):
        return self.__delta_weights

    def get_weights(self):
        return self.__weights

    def get_activation_function(self):
        return self.__activation_function

    def get_nodes(self):
        return self.__nodes

    def get_net(self):
        return self.__net

    def get_last_output(self):
        return self.__last_output
