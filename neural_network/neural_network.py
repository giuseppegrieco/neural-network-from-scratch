"""
    This module provide the concept of Artificial Neural Network.
"""
import numpy as np

from .utils import convert_in_numpy


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
                np.random.normal(0, 1 / np.sqrt(1 + previous_nodes), (layer.get_nodes(), 1 + previous_nodes))
            )
            layer.set_delta_old(
                np.zeros((layer.get_nodes(), 1 + previous_nodes))
            )
            previous_nodes = layer.get_nodes()
            layer.set_is_hidden(1)

        self.__topology[-1].set_is_hidden(0)

    def train_epochs_with_compare(self, input_data, expected_output, epochs, compare_data, compare_output):
        t_errors = []
        c_errors = []
        for i in range(0, epochs):
            t_errors.append(self.train(input_data, expected_output))
            c_errors.append(
                np.matrix.sum(np.power(compare_output - self.feed_forward(compare_data), 2)) * 1 / len(
                    np.mat(compare_output).T
                )
            )
        return t_errors, c_errors

    def train_epochs(self, input_data, expected_output, epochs):
        t_errors = []
        for i in range(0, epochs):
            t_errors.append(self.train(input_data, expected_output))
        return t_errors

    def train(self, input_data, expected_output):
        return self.__learning_algorithm.train(self, input_data, expected_output)

    def feed_forward(self, input_data):
        input_data = convert_in_numpy(input_data)

        for layer in self.__topology:
            input_data = layer.computes(input_data)

        return input_data

    def get_topology(self):
        return self.__topology
