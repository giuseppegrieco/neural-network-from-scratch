import datetime
import json
import os
import sys
import numpy as np

from neural_network.early_stopping import EarlyStoppingMinimalIncrease, EarlyStoppingValidationScore
from neural_network.functions import Sigmoid, Identity, MeanSquaredError
from neural_network.hyperparameter_tuning import GridSearch, CascadeCorrelationTuningSpecs, GradientDescentTuningSpecs
from neural_network.layers import Layer, RandomNormalInitializer
from neural_network.learning_algorithm import CascadeCorrelation
from neural_network.learning_observer import ErrorObserver
from neural_network.model_selection import KFoldCrossValidation

import matplotlib.pyplot as plt

from neural_network.neural_network_cc import NeuralNetworkCC


def create_output_json(eta, lambda_reg, alpha_momentum, epochs, duration_in_sec, topology, path, average, variance):
    data = \
        {
            'learning_algorithm':
                {  # TODO: fix it when the learning algorithm will be a parameter
                    'name': 'gradient_descent',
                    'learning_rate': eta,
                    'lambda_regularization': lambda_reg,
                    'alpha_momentum': alpha_momentum
                },
            'topology': {},
            'epochs': epochs,
            'duration_sec': duration_in_sec,
            'average': average,
            'variance': variance
        }

    index = 0
    for layer in topology:
        activation_function_name = layer.get_activation_function()
        activation_function_name = activation_function_name.__class__.__name__
        data['topology'][str(index)] = {'nodes': layer.get_nodes(), 'activation_function': activation_function_name}
        index = index + 1

    with open(path + 'data.json', 'w') as fp:
        json.dump(data, fp)


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
        regularization_correlation=0.01,
        regularization_pseudo_inverse=0.0006,
        activation_function=Sigmoid,
        weights_initializer=w_init,
        epochs=100,
        max_nodes=100,
        pool_size=10
    )
    e1 = ErrorObserver(neural_network=my_nn, X=X_train, Y=Y_train, error_function=MeanSquaredError)
    e2 = ErrorObserver(neural_network=my_nn, X=X_val, Y=Y_val, error_function=MeanSquaredError)
    cc.attach(e1)
    cc.attach(e2)
    cc.train(neural_network=my_nn, X_train=X_train, Y_train=Y_train)
    plotgraph(e1.store, e2.store)
    print(min(e2.store))
    sys.exit(0)


def create_timestamp_directory(path, prefix=''):  # todo: mettere secondo camnpo non obbligatiorio
    directory_name = path + prefix + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    try:
        os.mkdir(directory_name)
    except FileExistsError as e:
        raise e
    return directory_name

initial_path = ''

def save_grid_time(grid_search_duration_in_sec):
    data = {'duration_GS': grid_search_duration_in_sec}
    with open(initial_path + '/data.json', 'w') as fp:
        json.dump(data, fp)

def save_result(result):
    directory_name = create_timestamp_directory(initial_path + '/', "")

    with open(directory_name + '/hyperparameters.json', 'w') as fp:
        json.dump(result['hyperparameters'], fp)
    with open(directory_name + '/result.json', 'w') as fp:
        json.dump({
            'mean': result['result']['mean'],
            'variance': result['result']['variance']
        }, fp)
    for i in range(1, 6):
        fold_directory_name = directory_name + "/fold-" + str(i)
        try:
            os.mkdir(fold_directory_name)
        except FileExistsError as e:
            raise e
        fold = result['result'][str(i)]
        np.save(fold_directory_name + "/initial_weights", fold['initial_weights'])
        np.save(fold_directory_name + "/training_errors", fold['training_errors'])
        np.save(fold_directory_name + "/validation_errors", fold['validation_errors'])
        with open(fold_directory_name + '/result.json', 'w') as fp:
            json.dump({'validation_score': fold['validation_score']}, fp)


if __name__ == '__main__':
    start_time_GS = datetime.datetime.now().timestamp()

    w_init = RandomNormalInitializer()

    gds = CascadeCorrelationTuningSpecs(
        input_size=20,
        output_layer_list=[Layer(2, Identity, w_init)],
        learning_rate_list=[0.0005, 0.0001, 0.00001, 0.000001, 0.0000001],
        momentum_list=[0.9, 0.6, 0.3, 0.1, 0],
        regularization_correlation_list=[0.00001, 0.000001],
        regularization_pseudo_inverse_list=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
        max_nodes_list=[300],
        pool_size_list=[30],
        epochs_list=[15000],
        weights_initializer_list=[w_init],
        activation_function_list=[Sigmoid],
        minimal_correlation_increase_list=[0.001],
        max_fails_increase_list=[50]
    )



    TS = np.genfromtxt('cup/tr.csv', delimiter=',')
    TS = TS[:, 1:]

    X_train = TS[:, :-2].T
    Y_train = TS[:, -2:].T

    cross_validation = KFoldCrossValidation(5, MeanSquaredError)
    cross_validation.add_early_stopping(
        EarlyStoppingMinimalIncrease(0.00001, 20)
    )
    cross_validation.add_early_stopping(
        EarlyStoppingValidationScore(10)
    )

    gs = GridSearch(gds, cross_validation)

    initial_path = create_timestamp_directory("./grid_search/", "GS-")

    grid_result = gs.run(2, X_train, Y_train, save_result)

    end_time_GS = datetime.datetime.now().timestamp()
    grid_search_duration_in_sec = end_time_GS - start_time_GS
    save_grid_time(grid_search_duration_in_sec)

    sys.exit(0)
