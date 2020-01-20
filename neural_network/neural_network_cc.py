import numpy as np

from neural_network import NeuralNetwork
from neural_network.layers import Layer


class NeuralNetworkCC(NeuralNetwork):
    def pack(self):
        output_layer = self._layers[-1]
        output_layer.initialize_weights(self._input_size)
        self._layers.clear()
        self._layers.append(output_layer)

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