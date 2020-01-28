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
        """
        This is the constructor for the class Layer.

        :param number_of_nodes: int
        :param activation_function: Function class
        :param weights_initializer: WeightsInitializer
        """
        self.activation_function = activation_function
        self.number_of_nodes = number_of_nodes
        self.__weights_initializer = weights_initializer

        self.weights = None
        self.last_input = None
        self.net = None
        self.is_hidden = False

    def computes(self, input_data):
        """
        This method performs the net of the layer and apply the activation function.

        :param input_data: np.mat
        :return: np.mat
        """
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
        """
        This method add a row of  1 to the input indicated as parameter,
        this is done to multiply for the bias.

        :param target: np.mat

        :return: np.mat
        """
        n, m = target.shape
        bias = np.ones((1, m), dtype=np.dtype('d'))
        return np.vstack((bias, target))

    def initialize_weights(self, in_size):
        """
        This methods call the weights initializer to initializes the weights of the layer

        :param in_size: int
        """
        self.__weights_initializer.initializes(self, (self.__number_of_nodes, in_size + 1))

    @property
    def activation_function(self):
        """
        It returns the activation function.

        :return: Function class
        """
        return self.__activation_function

    @activation_function.setter
    def activation_function(self, activation_function):
        """
        Allows to set the activation function.

        :param activation_function: Function class
        """
        self.__activation_function = activation_function

    @property
    def number_of_nodes(self) -> int:
        """
        It returns number of nodes.

        :return: int
        """
        return self.__number_of_nodes

    @number_of_nodes.setter
    def number_of_nodes(self, number_of_nodes: int):
        """
        Allows to set the number of nodes.

        :param number_of_nodes: int
        """
        self.__number_of_nodes = number_of_nodes

    @property
    def weights(self) -> np.mat:
        """
        It returns the weights.

        :return: np.mat
        """
        return self.__weights

    @weights.setter
    def weights(self, weights: np.mat):
        """
        Allows to set the weights.

        :param weights: np.mat
        """
        self.__weights = weights

    @property
    def net(self) -> np.mat:
        """
        It returns the net.

        :return: np.mat
        """
        return self.__net

    @net.setter
    def net(self, net: np.mat):
        """
        Allows to set the net.

        :param net: np.mat
        """
        self.__net = net

    @property
    def last_input(self) -> np.mat:
        """
        It returns last input.

        :return: np.mat
        """
        return self.__last_input

    @last_input.setter
    def last_input(self, last_input: np.mat):
        """
        Allows to set the last input.

        :param last_input: np.mat
        """
        self.__last_input = last_input

    @property
    def is_hidden(self) -> bool:
        """
        It indicates if the layer is a hidden layer.

        :return: boolean
        """
        return self.__is_hidden

    @is_hidden.setter
    def is_hidden(self, is_hidden: bool):
        """
        Allows to set the layer as a hidden layer.

        :param is_hidden: bool
        """
        self.__is_hidden = is_hidden

    def __repr__(self):
        return "number_of_nodes=%d, activation_function=%s" % \
               (self.__number_of_nodes, str(self.__activation_function.__name__))
