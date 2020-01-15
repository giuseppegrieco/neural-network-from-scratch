"""
This module provides the concept of Artificial Neural Network.
"""
import numpy as np

from .utils import convert_in_numpy


class NeuralNetwork:
    """
    This class is an abstraction of Artificial Neural Network concept,
    encapsulates the topology and the specifics on learning and provides
    a public interface to initializes and query the network.
    """
    def __init__(self,
                 input_size,
                 topology):
        """
        Constructor

        Args:
            input_size: number of features of a single input
            topology: set of layers
        """
        # Checks the correctness of parameter: input_size
        try:
            float(input_size)
            if input_size <= 0:
                raise ValueError('task parameter must be greater than 0')
        except ValueError:
            raise ValueError('task parameter must be a float')
        self.__input_size = input_size

        # TODO: Check activation function topology
        self.__topology = topology

        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of network layers
        """
        previous_nodes = self.__input_size

        for layer in self.__topology:
            layer.set_weights(
                np.random.normal(0, 1 / np.sqrt(1 + previous_nodes), (layer.get_nodes(), 1 + previous_nodes))
            )
            layer.set_weights(
                layer.get_weights().astype(np.dtype('d'))
            )
            layer.set_delta_old(
                np.zeros((layer.get_nodes(), 1 + previous_nodes), dtype=np.dtype('d'))
            )
            previous_nodes = layer.get_nodes()
            layer.set_is_hidden(1)

        self.__topology[-1].set_is_hidden(0)

    def feed_forward(self, input_data):
        """
        Performs feed forward with data indicated

        Args:
            input_data: input in matrix form

        Return:
            prediction of the network for the data indicated
        """
        input_data = convert_in_numpy(input_data)

        for layer in self.__topology:
            input_data = layer.computes(input_data)

        return input_data

    def get_topology(self):
        """
        Returns the topology of the network

        Returns:
            topology
        """
        return self.__topology

    def get_all_weights(self):
        """
        Returns all weights of network layers

        Returns:
             weights
        """
        weights = []
        for layer in self.get_topology():
            weights.append(layer.get_weights())
        return weights

    def get_number_of_nodes(self):
        """
        Return a vector with in position i the nodes number of i-th layer

        Returns:
            vector of nodes number
        """
        nodes = []
        for layer in self.get_topology():
            nodes.append(layer.get_nodes())
        return nodes
