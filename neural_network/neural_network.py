from typing import List

from neural_network.layers import Layer


class NeuralNetwork(object):
    def __init__(self, input_size: int, layers: List[Layer]):
        """
        This is the constructor for the class Neural Network, and is used
        to specify input size and topology.

        :param input_size: int
        :param layers: List[Layer]
        """
        self._input_size = input_size
        self.layers = layers
        self.pack()

    def pack(self):
        """
        Performs weights initialization of all layer.
        """
        in_size = self._input_size
        for layer in self._layers:
            layer.is_hidden = True
            layer.initialize_weights(in_size)
            in_size = layer.number_of_nodes
        self._layers[-1].is_hidden = False

    def feed_forward(self, X_train):
        """
        Predicts a value given an input, by performing feed forward.

        :param X_train: input
        :return: predicted output for input given
        """
        output = Layer.add_bias_input(X_train)
        for layer in self._layers:
            output = layer.computes(output)
        return output

    @property
    def layers(self) -> List[Layer]:
        """
        Returns the list of network's layers.

        :return: List[Layer]
        """
        return self._layers

    @layers.setter
    def layers(self, layers: List[Layer]):
        """
        Allows to set a list of layers for the classes.

        :param layers: List[Layer]
        """
        self._layers = layers
