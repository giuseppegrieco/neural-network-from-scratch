"""
This module provides the concept of Neural Network's Full-connected-layer
"""
from .utils import convert_in_numpy


class Layer:
    """
    This class is an abstraction of Neural Network's full-connected-layer
    concept, encapsulates the state of a layer and provides a public interface
    to retrieves the output of the layer
    """
    def __init__(self, nodes, activation_function):
        """
        Constructor

        Args:
            nodes: number of layer nodes
            activation_function: layer activation function
        """

        # Checks the correctness of parameter: nodes
        try:
            int(nodes)
            if nodes <= 0:
                raise ValueError('nodes parameter must be greater than 0')
        except ValueError:
            raise ValueError('nodes parameter must be an integer')
        self.__nodes = nodes

        self.__activation_function = activation_function

        # Initialization
        self.__weights = None
        self.__net = None
        self.__last_output = None
        self.__is_hidden = None
        self.__delta_old = None

    def computes(self, input_data):
        """
        Computes the output of the layer

        Args:
            input_data: layer input in matrix form

        Returns:
            the output of the layer
        """
        # performs the net of the layer
        self.__net = self.__weights.dot(
            input_data
        )

        # calculates the output of the layer
        self.__last_output = self.__activation_function(self.__net)
        if self.__is_hidden:
            self.__last_output = convert_in_numpy(self.__last_output)
        return self.__last_output

    def set_weights(self, weights):
        """
        Set new weights of the layer

        Args:
             weights: the new weights of the layer
        """
        self.__weights = weights

    def set_delta_old(self, delta_old):
        """
        Set new delta old of the layer

        Args:
             delta_old: the new delta old of the layer
        """
        self.__delta_old = delta_old

    def set_is_hidden(self, is_hidden):
        """
        Set if the layer is an hidden layer or not

        Args:
             is_hidden: true if is an hidden layer, false otherwise
        """
        self.__is_hidden = is_hidden

    def get_is_hidden(self):
        """
        Indicates if the layer is an hidden layer or not

        Returns:
            true if is an hidden layer, false otherwise
        """
        return self.__is_hidden

    def get_delta_old(self):
        """
        Returns the delta old of the layer

        Returns:
            delta old
        """
        return self.__delta_old

    def get_weights(self):
        """
        Returns the weights of the layer

        Returns:
            weights
        """
        return self.__weights

    def get_activation_function(self):
        """
        Returns the activation function of the layer

        Returns:
            activation function
        """
        return self.__activation_function

    def get_nodes(self):
        """
        Returns the number of layer nodes

        Returns:
            number of layer nodes
        """
        return self.__nodes

    def get_net(self):
        """
        Returns the net of the layer

        Returns:
            net of the layer
        """
        return self.__net

    def get_last_output(self):
        """
        Returns the last output of the layer

        Returns:
            last output of the layer
        """
        return self.__last_output
