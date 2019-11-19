"""
    This module provide the concept of Neural Network's Layer.
"""
import numpy as np

from utils import convert_in_numpy


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
        self.__delta_old = None

    def computes(self, input_data):
        # performs the net of the layer
        self.__net = self.__weights.dot(
            input_data
        )

        # calculates the output of the layer
        self.__last_output = self.__activation_function.f(self.__net)
        if self.__is_hidden:
            self.__last_output = convert_in_numpy(self.__last_output)
        return self.__last_output

    def set_weights(self, weights):
        self.__weights = weights

    def set_delta_old(self, delta_old):
        self.__delta_old = delta_old

    def set_is_hidden(self, is_hidden):
        self.__is_hidden = is_hidden

    def get_delta_old(self):
        return self.__delta_old

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
