from typing import List

from neural_network.layers import Layer


class NeuralNetwork(object):
    def __init__(self, input_size: int, layers: List[Layer]):
        self.__input_size = input_size
        self.layers = layers
        self.pack()

    def pack(self):
        in_size = self.__input_size
        for layer in self._layers:
            layer.is_hidden = True
            layer.initialize_weights(in_size)
            in_size = layer.number_of_nodes
        self._layers[-1].is_hidden = False

    def feed_forward(self, X_train):
        output = Layer.add_bias_input(X_train)
        for layer in self._layers:
            output = layer.computes(output)
        return output

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    @layers.setter
    def layers(self, layers: List[Layer]):
        self._layers = layers
