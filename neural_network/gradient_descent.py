"""
This module provides an implementation for gradient descent algorithm.
"""
import numpy as np

from .utils import convert_in_numpy


def gradient_descent(
        neural_network,
        input_data,
        expected_output,
        learning_rate,
        lambda_regularization,
        alpha_momentum
):
    """
    Performs gradient descent learning algorithm with settings
    in according to the hyperparameters specified.

    Params:
        neural_network: network where apply the algorithm
        input_data: input data in matrix form
        expected_output: expected output of input data, indicated in matrix form
        learning_rate: coefficient to control the weights change w.r.t error each epoch
        lambda_regularization: coefficient to be used in the regularization
        alpha_momentum: coefficient to be used in the momentum
    """
    # Checks the correctness of parameter: learning_rate
    try:
        float(learning_rate)
        if learning_rate <= 0:
            raise ValueError('learning_rate parameter must be greater than 0')
    except ValueError:
        raise ValueError('learning_rate parameter must be a float')

    # Checks the correctness of parameter: lambda_regularization
    try:
        float(lambda_regularization)
        if lambda_regularization < 0:
            raise ValueError('lambda_regularization parameter must be greater or equals than 0')
    except ValueError:
        raise ValueError('lambda_regularization parameter must be a float')

    # Checks the correctness of parameter: alpha_momentum
    try:
        float(alpha_momentum)
        if alpha_momentum < 0:
            raise ValueError('alpha_momentum parameter must be greater or equals than 0')
    except ValueError:
        raise ValueError('alpha_momentum parameter must be a float')

    for layer in neural_network.get_topology():
        layer.set_weights(
            layer.get_weights() + (alpha_momentum * layer.get_delta_old())
        )

    output = neural_network.feed_forward(input_data)

    input_data = convert_in_numpy(input_data)

    # reshape target vector as column vector
    expected_output = np.mat(
        np.array(expected_output),
        dtype=np.dtype('d')
    )

    __back_propagation(
        neural_network,
        input_data,
        expected_output,
        output,
        learning_rate,
        lambda_regularization,
        alpha_momentum
    )
    return np.matrix.mean(np.power(expected_output - output, 2))


def __back_propagation(
        neural_network,
        input_data,
        target,
        output,
        learning_rate,
        lambda_regularization,
        alpha_momentum):
    """
    Performs back-propagation.

    TODO: fix the method (note output is already vector column)
    """
    output_layer = neural_network.get_topology()[-1]
    hidden_layers = neural_network.get_topology()[:-1]
    first_hidden_layer = hidden_layers[-1]

    # error of each output nodes
    delta = (target - output)

    # TODO add formula as comment
    delta = np.multiply(delta, output_layer.get_activation_function()(
        output_layer.get_net(),
        derivative=True
    ))
    # TODO add formula as comment
    delta_oh = delta * first_hidden_layer.get_last_output().T

    # store previous weights
    previous_weights = output_layer.get_weights()

    # adjusting weights between last hidden layer and the output layer
    __adjusting_weights(
        output_layer,
        delta_oh,
        learning_rate,
        lambda_regularization,
        alpha_momentum
    )

    for hidden_layer_index in range(len(hidden_layers) - 1, 0, -1):
        # TODO add formula as comment
        delta = delta.T * previous_weights

        # remove bias, TODO explain why
        delta = delta[:, 1:]

        # TODO add formula as comment
        delta = np.multiply(delta.T, hidden_layers[hidden_layer_index].get_activation_function()(
            hidden_layers[hidden_layer_index].get_net(),
            derivative=True
        ))

        # TODO add formula as comment
        delta_hh = delta * hidden_layers[hidden_layer_index - 1].get_last_output().T

        # TODO add formula as comment
        previous_weights = hidden_layers[hidden_layer_index].get_weights()

        # adjusting weights between two hidden layers
        __adjusting_weights(
            hidden_layers[hidden_layer_index],
            delta_hh,
            learning_rate,
            lambda_regularization,
            alpha_momentum
        )

    # TODO add formula as comment
    delta = delta.T * previous_weights

    # remove bias, TODO explain why
    delta = delta[:, 1:]

    # TODO add formula as comment
    delta = np.multiply(delta.T, hidden_layers[0].get_activation_function()(
        hidden_layers[0].get_net(),
        derivative=True
    ))

    # TODO add formula as comment
    delta_hi = delta * input_data.T

    # adjusting weights between last hidden layer and the output layer
    __adjusting_weights(
        hidden_layers[0],
        delta_hi,
        learning_rate,
        lambda_regularization,
        alpha_momentum
    )


def __adjusting_weights(
        layer,
        delta,
        learning_rate,
        lambda_regularization,
        alpha_momentum):
    # retrieves current layer weights
    layer_weights = layer.get_weights()

    # prepare matrix multiplication of λ to apply regularization
    lambda_mat = np.full(layer_weights.shape, -lambda_regularization, dtype=np.dtype('d'))
    if layer.get_is_hidden():
        lambda_mat[:, 0] = 0.0  # exclude the bias from regularization

    # (-η * Δw) + (-λ * w) + (-α * Δw_old)
    delta_layer = (learning_rate * delta) + \
                  (alpha_momentum * layer.get_delta_old()) + \
                  np.multiply(lambda_mat, layer_weights)

    # update weights in according to delta rule
    layer.set_weights(
        layer_weights +
        delta_layer
    )
    layer.set_delta_old(delta_layer)
