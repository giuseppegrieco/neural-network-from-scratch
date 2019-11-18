"""
    This module provide the concept of Neural Network's Layer.
"""


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

    def computes(self, input_data):
        input_data = input_data.reshape((len(input_data), 1))

        # performs the net of the layer
        self.__net = self.__weights.dot(
            input_data
        )

        # calculates the output of the layer
        self.__last_output = self.__activation_function.f(self.__net)

        return self.__last_output

    def set_bias(self, bias):
        self.__bias = bias

    def set_weights(self, weights):
        self.__weights = weights

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
