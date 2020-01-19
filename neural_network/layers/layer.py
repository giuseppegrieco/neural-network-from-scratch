from __future__ import annotations

import numpy as np

import neural_network.layers as layers


class Layer(object):

    def __init__(
            self,
            number_of_nodes: int,
            activation_function,
            weights_initializer: layers.WeightsInitializer
    ):
        self.activation_function = activation_function
        self.number_of_nodes = number_of_nodes
        self.__weights_initializer = weights_initializer

        self.weights = None
        self.last_input = None
        self.net = None
        self.is_hidden = False

    def computes(self, input_data):
        self.__last_input = input_data
        self.__net = self.weights.dot(input_data)

        output = self.__activation_function.evaluate(
            self.__net
        )

        if self.__is_hidden:
            output = self.add_bias_input(output)

        return output

    @staticmethod
    def add_bias_input(target):
        bias = np.ones((1, len(target.T)))
        return np.vstack((target, bias))

    def initialize_weights(self, in_size):
        self.__weights_initializer.initializes(self, (self.__number_of_nodes, in_size + 1))

    @property
    def activation_function(self):
        return self.__activation_function

    @activation_function.setter
    def activation_function(self, activation_function):
        self.__activation_function = activation_function

    @property
    def number_of_nodes(self) -> int:
        return self.__number_of_nodes

    @number_of_nodes.setter
    def number_of_nodes(self, number_of_nodes: int):
        self.__number_of_nodes = number_of_nodes

    @property
    def weights(self) -> np.mat:
        return self.__weights

    @weights.setter
    def weights(self, weights: np.mat):
        self.__weights = weights

    @property
    def net(self) -> np.mat:
        return self.__net

    @net.setter
    def net(self, net: np.mat):
        self.__net = net

    @property
    def last_input(self) -> np.mat:
        return self.__last_input

    @last_input.setter
    def last_input(self, last_input: np.mat):
        self.__last_input = last_input

    @property
    def is_hidden(self) -> bool:
        return self.__is_hidden

    @is_hidden.setter
    def is_hidden(self, is_hidden: bool):
        self.__is_hidden = is_hidden

    def __repr__(self):
        return "number_of_nodes=%d, activation_function=%s" % \
               (self.__number_of_nodes, str(self.__activation_function.__name__))
