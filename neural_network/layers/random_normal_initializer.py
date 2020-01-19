import numpy as np

from neural_network.layers import WeightsInitializer
from neural_network.layers.layer import Layer


class RandomNormalInitializer(WeightsInitializer):
    def initializes(self, layer: Layer, shapes):
        layer.weights = np.random.normal(
            np.random.normal(0, 1 / np.sqrt(shapes[1]), shapes)
        )
