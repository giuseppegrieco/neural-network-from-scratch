from abc import ABC, abstractmethod

from neural_network.layers.layer import Layer


class WeightsInitializer(ABC):
    @abstractmethod
    def initializes(self, layer: Layer, shapes):
        """
        This method specify the behavior of the weights initializer.

        :param layer: Layer
        :param shapes: Tuple
        """
        pass
