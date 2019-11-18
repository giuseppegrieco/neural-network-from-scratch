"""
    This module provide the concept of Artificial Neural Network.
"""
import numpy as np


class NeuralNetwork:
    def __init__(self,
                 input_size,
                 topology,
                 learning_algorithm):
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

        # TODO: Check activation function learning algorithm
        self.__learning_algorithm = learning_algorithm

        self.__init_weights()

    def __init_weights(self):
        previous_nodes = self.__input_size

        for layer in self.__topology:
            layer.set_weights(
                np.random.rand(layer.get_nodes(), previous_nodes + 1)
            )
            previous_nodes = layer.get_nodes()

    def train(self, input_data, expected_output):
        return self.__learning_algorithm.train(self, input_data, expected_output)

    def feed_forward(self, input_data):
        input_data = np.array(input_data, dtype=float)
        for layer in self.__topology:
            input_data = layer.computes(input_data)
        return input_data

    def get_topology(self):
        return self.__topology
