from neural_network.activation_function import Sigmoid, Identity
from neural_network.layer import Layer

import numpy as np
import sys

from neural_network.loss import MSE


def CC(
        neural_network,
        training_data,
        activation_function,
        epochs,
        learning_rate,
        momentum,
        regularization
):
    tr_input, tr_output = training_data

    n, m = tr_input.shape
    bias = np.ones((1, m), dtype=np.dtype('d'))
    tr_input = np.vstack((tr_input, bias))

    output_layer = CCOutputLayer(tr_input, tr_output, regularization, Identity())
    neural_network.set_output_layer(output_layer)

    previous_error = sys.float_info.min
    current_epoch = 0
    last_output = tr_input

    # TODO WE HAVE ALREADY COMPUTED
    new_error = MSE().evaluate(output_layer.get_previous_output(), tr_output)

    print("Initial error : % 5.10f" %(new_error))
    i = 0
    while current_epoch < epochs + 1:
        print("------------------- Node : % 2d ------------------- " %(current_epoch + 1))
        previous_error = new_error

        hidden_layer = CCHiddenLayer(
            Sigmoid(),
            last_output
        )

        last_predicted_value = output_layer.get_previous_output()

        E = np.power(tr_output - last_predicted_value, 2)
        E_mean = np.sum(E, axis=1) * 1 / len(tr_input.T)
        E_mean = np.reshape(E_mean, (len(tr_output), 1))
        i += 1
        E_tot = E - E_mean

        new_correlation, V_tot = __calculates_correlation(
            hidden_layer,
            E_tot
        )
        correlation = sys.float_info.min
        delta_old = np.zeros(hidden_layer.get_weights().shape)
        i = 10
        while i > 0:
            if new_correlation - correlation < correlation * 0.001:
                i = i - 1
            else:
                i = 20
            correlation = new_correlation

            delta = np.sign(
                __calculates_output_correlation(V_tot, E_tot)
            )
            delta = np.reshape(delta, (len(tr_output), 1))
            delta = np.multiply(delta, E - E_mean)
            delta = np.multiply(delta, hidden_layer.get_activation_function().d(
                hidden_layer.get_weights().T.dot(hidden_layer.get_previous_output())
            ))
            delta = np.sum(np.dot(delta, hidden_layer.get_previous_output().T).T, axis=1)
            delta = np.reshape(delta, (len(last_output), 1))

            hidden_layer.set_weights(
                hidden_layer.get_weights() +
                (learning_rate * delta) +
                (momentum * delta_old) +
                (regularization * -hidden_layer.get_weights())
            )
            delta_old = delta
            new_correlation, V_tot = __calculates_correlation(
                hidden_layer,
                E_tot
            )
            print(new_correlation)


        output_layer.add_weights(
            hidden_layer.computes_optimized_for_training(),
            tr_output
        )
        neural_network.add_hidden_layer(
            hidden_layer
        )

        new_error = MSE().evaluate(
            output_layer.get_previous_output(),
            tr_output
        )
        print("------------------- MSE : % 5.10f ------------------- " %(new_error))
        last_output = np.vstack((
            last_output,
            hidden_layer.computes_optimized_for_training()
        ))
        current_epoch += 1
        print("----------------------------------------------------- " %(new_error))


def __calculates_correlation(hidden_layer, E_tot):
    V = hidden_layer.computes_optimized_for_training()
    print('dio')
    print(V.shape)
    V_mean = np.mean(V)
    V_tot = V - V_mean

    return np.sum(
        np.absolute(__calculates_output_correlation(V_tot, E_tot)),
        axis=0
    ), V_tot


def __calculates_output_correlation(V_tot, E_tot):
    return np.sum(
        np.multiply(V_tot, E_tot),
        axis=1
    )


def __calculate_error():
    pass


class CCHiddenLayer(Layer):
    def __init__(self, activation_function, previous_output):
        self.__activation_function = activation_function
        self.__previous_output = previous_output

        self.__weights = np.random.normal(
            0, 1 / np.sqrt(len(self.__previous_output)),
            (len(self.__previous_output), 1)
        )

    def get_weights(self):
        return self.__weights

    def set_weights(self, weights):
        self.__weights = weights

    def get_activation_function(self):
        return self.__activation_function

    def computes_optimized_for_training(self):
        return self.computes(self.__previous_output)

    def get_previous_output(self):
        return self.__previous_output

    def computes(self, input_data):
        return self.__activation_function.f(
            self.__weights.T.dot(input_data)
        )


class CCOutputLayer(Layer):
    def __init__(self, input_data, expected_output, regularization, activation_function):
        self.__weights = None
        self.__last_output = 0
        self.__regularization = regularization
        self.__activation_function = activation_function

        self.add_weights(input_data, expected_output)

    def add_weights(self, input_data, expected_output):
        psuedo = self.__pseudo_inverse(input_data).T
        new_weights = np.dot(
            (expected_output - self.__last_output),
            psuedo
        )
        print('new:')
        print(new_weights)
        self.__last_output = self.__last_output + new_weights.dot(
            input_data
        )
        if self.__weights is None:
            self.__weights = new_weights
        else:
            self.__weights = np.hstack((
                self.__weights,
                new_weights
            ))

    def get_activation_function(self):
        return self.__activation_function

    def __pseudo_inverse(self, input_data):
        return np.dot(np.linalg.inv(
            np.dot(
                input_data,
                input_data.T
            ) + (np.identity(len(input_data)) * -self.__regularization)
        ), input_data)

    def get_previous_output(self):
        return self.__last_output

    def computes(self, input_data):
        return self.__activation_function.f(
            self.__weights.T.dot(input_data)
        )