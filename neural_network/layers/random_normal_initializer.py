import numpy as np

from neural_network.layers import WeightsInitializer
from neural_network.layers.layer import Layer


class RandomNormalInitializer(WeightsInitializer):
    def initializes(self, layer: Layer, shapes):
        """
        This method initializes the weight with normal distribution with 0 mean
        and standard devation 1 / sqrt(incoming connections).

        :param layer: Layer
        :param shapes: Tuple
        """
        layer.weights = np.random.uniform(0, 1 / np.sqrt(shapes[1]), shapes)
