import numpy as np


class _Layer:
    def __init__(self, previous_nodes, nodes, activation_function, activation_function_derivative):
        self.__activation_function = activation_function
        self.__activation_function_derivative = activation_function_derivative;
        self.__weights = np.random.rand(nodes, 1 + previous_nodes)
        self.__last_output = None
        self.__net = None

    def computes(self, inputs):
        inputs = np.append(np.array([1]), inputs)
        self.__net = self.__weights.dot(
            inputs
        )
        self.__last_output = self.__activation_function(self.__net)
        return self.__last_output

    def get_activation_function_derivative(self):
        return self.__activation_function_derivative

    def get_net(self):
        return self.__net

    def get_node_weights(self, node):
        return self.__weights[node]

    def set_node_weights(self, node, weights):
        self.__weights[node] = weights

    def get_weights(self):
        return self.__weights

    def set_weights(self, weights):
        self.__weights = weights

    def get_last_output(self):
        return self.__last_output

    def get_activation_function(self):
        return self.__activation_function

    def set_activation_function(self, activation_function):
        self.__activation_function = activation_function

    def set_activation_function_derivative(self, activation_function_derivative):
        self.__activation_function_derivative = activation_function_derivative
