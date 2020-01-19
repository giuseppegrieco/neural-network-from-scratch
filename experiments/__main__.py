import sys
import numpy as np

from neural_network import NeuralNetwork
from neural_network.functions import Sigmoid, Identity, MeanSquaredError
from neural_network.hyperparameter_tuning import GradientDescentTuningSpecs, GridSearch
from neural_network.layers import Layer, RandomNormalInitializer
from neural_network.learning_algorithm import GradientDescent, CascadeCorrelation
from neural_network.learning_observer import ErrorObserver
from neural_network.model_selection import KFoldCrossValidation

import matplotlib.pyplot as plt

from neural_network.neural_network_cc import NeuralNetworkCC


def plotgraph(training_errors, validation_errors):
    fig, subplot = plt.subplots(nrows=1, ncols=1)
    subplot.plot(training_errors, '-r', label='Training')
    subplot.plot(validation_errors, '--g', label='Validation')
    subplot.legend()
    plt.show()

    print(min(validation_errors))


def cascade():
    w_init = RandomNormalInitializer()
    TS = np.genfromtxt('cup/tr.csv', delimiter=',')

    TS = TS[:, 1:]

    X_train = TS[:1000, :-2].T
    Y_train = TS[:1000, -2:].T

    X_val = TS[1000:, :-2].T
    Y_val = TS[1000:, -2:].T

    my_nn = NeuralNetworkCC(
        20,
        [
            Layer(2, Identity, w_init)
        ]
    )
    cc = CascadeCorrelation(
        learning_rate=0.4,
        momentum=0.6,
        regularization_correlation=0.0001,
        regularization_pseudo_inverse=0.0001,
        activation_function=Sigmoid,
        weights_initializer=w_init,
        epochs=30000,
        max_nodes=100
    )
    e1 = ErrorObserver(neural_network=my_nn, X=X_train, Y=Y_train, error_function=MeanSquaredError)
    e2 = ErrorObserver(neural_network=my_nn, X=X_val, Y=Y_val, error_function=MeanSquaredError)
    cc.attach(e1)
    cc.attach(e2)
    cc.train(neural_network=my_nn, X_train=X_train, Y_train=Y_train)
    plotgraph(e1.store, e2.store)
    print(min(e2.store))
    sys.exit(0)


def print_mean(mean, variance):
    print('mean')
    print(mean)
    print('variance')
    print(variance)


if __name__ == '__main__':
    cascade()
    w_init = RandomNormalInitializer()

    gds = GradientDescentTuningSpecs(
        input_size=20,
        layers_list=[
            [
                Layer(225, Sigmoid, w_init),
                Layer(2, Identity, w_init)
            ]
        ],
        learning_rate_list=[
            6e-05
        ],
        momentum_list=[
            0.6
        ],
        regularization_list=[
            4e-05
        ],
        epochs=15000
    )

    TS = np.genfromtxt('cup/tr.csv', delimiter=',')
    TS = TS[:, 1:]

    X_train = TS[:, :-2].T
    Y_train = TS[:, -2:].T

    cross_validation = KFoldCrossValidation(5, MeanSquaredError)
    cross_validation.on_finish = print_mean

    gs = GridSearch(gds, cross_validation)

    grid_result = gs.run(2, X_train, Y_train)

    sys.exit(0)
