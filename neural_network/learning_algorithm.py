import numpy as np

from .utils import convert_in_numpy


class LearningAlgorithm:
    def train(self, neural_network, input_data, expected_output):
        pass


class GradientDescent(LearningAlgorithm):
    def __init__(self, learning_rate, lambda_regularization, alpha_momentum):
        # Checks the correctness of parameter: learning_rate
        try:
            float(learning_rate)
            if learning_rate <= 0:
                raise ValueError('learning_rate parameter must be greater than 0')
        except ValueError:
            raise ValueError('learning_rate parameter must be a float')
        self.__learning_rate = learning_rate

        # Checks the correctness of parameter: lambda_regularization
        try:
            float(lambda_regularization)
            if lambda_regularization < 0:
                raise ValueError('lambda_regularization parameter must be greater or equals than 0')
        except ValueError:
            raise ValueError('lambda_regularization parameter must be a float')
        self.__lambda_regularization = lambda_regularization

        # Checks the correctness of parameter: alpha_momentum
        try:
            float(alpha_momentum)
            if alpha_momentum < 0:
                raise ValueError('alpha_momentum parameter must be greater or equals than 0')
        except ValueError:
            raise ValueError('alpha_momentum parameter must be a float')
        self.__alpha_momentum = alpha_momentum

    def train(self, neural_network, input_data, expected_output):
        output = neural_network.feed_forward(input_data)
        input_data = convert_in_numpy(input_data)

        # reshape target vector as column vector

        expected_output = np.mat(np.array(expected_output), dtype=np.dtype('d'))

        self.__back_propagation(
            neural_network,
            input_data,
            expected_output,
            output
        )
        return np.matrix.sum(np.power(expected_output - output, 2)) * (1 / len(expected_output.T))

    def __back_propagation(self, neural_network, input_data, target, output):
        """
        Performs back-propagation.

        TODO: fix the method (note output is already vector column)
        """

        output_layer = neural_network.get_topology()[-1]
        hidden_layers = neural_network.get_topology()[:-1]
        first_hidden_layer = hidden_layers[-1]

        # error of each output nodes
        delta = (target - output)

        delta = np.multiply(delta, output_layer.get_activation_function()(
            output_layer.get_net(),
            derivative=True
        ))

        # adjusting delta of weights between last hidden layer and the output layer
        delta_oh = delta * first_hidden_layer.get_last_output().T

        previous_weights = output_layer.get_weights()

        self.__adjusting_weights(output_layer, delta_oh)

        for hidden_layer_index in range(len(hidden_layers) - 1, 0, -1):
            delta = delta.T * previous_weights
            delta = delta[:, 1:]
            delta = np.multiply(delta.T, hidden_layers[hidden_layer_index].get_activation_function()(
                hidden_layers[hidden_layer_index].get_net(),
                derivative=True
            ))

            # adjusting delta of weights between two hidden layers
            delta_hh = delta * hidden_layers[hidden_layer_index - 1].get_last_output().T
            previous_weights = hidden_layers[hidden_layer_index].get_weights()
            self.__adjusting_weights(hidden_layers[hidden_layer_index], delta_hh)

        delta = delta.T * previous_weights
        delta = delta[:, 1:]
        delta = np.multiply(delta.T, hidden_layers[0].get_activation_function()(
            hidden_layers[0].get_net(),
            derivative=True
        ))

        # adjusting delta of weights between last hidden layer and the output layer
        delta_hi = delta * input_data.T
        self.__adjusting_weights(hidden_layers[0], delta_hi)

    def __adjusting_weights(self, layer, delta):
        # (-η * Δw) + (-λ * w) + (-α * Δw_old)
        delta_layer = (self.__learning_rate * delta) +\
                      (self.__alpha_momentum * layer.get_delta_old())
        layer_weights = layer.get_weights()
        lambda_mat = np.full(layer_weights.shape, -self.__lambda_regularization, dtype=np.dtype('d'))
        if layer.get_is_hidden():
            lambda_mat[:, 0] = 0.0
        delta_layer = delta_layer + np.multiply(lambda_mat, layer_weights)
        layer.set_weights(
            layer_weights +   # w (old weights)
            delta_layer
        )
        layer.set_delta_old(delta_layer)
