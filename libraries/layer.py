import numpy as np


class _Layer:
    def __init__(self, previous_nodes, nodes, activation_function):
        self.__activation_function = activation_function
        self.__weights = np.random.rand(nodes, previous_nodes)

    def get_node_weights(self, node):
        return self.__weights[node]

    def set_node_weights(self, node, weights):
        self.__weights[node] = weights

    def get_weights(self):
        return self.__weights

    def computes(self, inputs):
        return self.__activation_function(self.__weights.dot(inputs))

    def get_activation_function(self):
        return self.__activation_function

    def set_activation_function(self, activation_function):
        self.__activation_function = activation_function
