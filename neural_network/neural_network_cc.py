import numpy as np

from neural_network import NeuralNetwork
from neural_network.layers import Layer


class NeuralNetworkCC(NeuralNetwork):
    def pack(self):
        self._layers = [self._layers[-1]]
        super().pack()

    def feed_forward(self, X_train):
        current_input = Layer.add_bias_input(X_train)
        output = None
        for layer in self._layers:
            output = layer.computes(current_input)
            current_input = np.vstack((
                current_input,
                output
            ))
        return output