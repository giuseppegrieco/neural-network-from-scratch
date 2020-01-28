import csv
import datetime
import json
import logging
import os
import pickle
import sys
import numpy as np

from neural_network.early_stopping import EarlyStoppingMinimalIncrease, EarlyStoppingValidationScore
from neural_network.functions import Sigmoid, Identity, MeanSquaredError, MeanEuclideanError
from neural_network.hyperparameter_tuning import GridSearch, CascadeCorrelationTuningSpecs
from neural_network.layers import Layer, RandomNormalInitializer
from neural_network.model_selection import KFoldCrossValidation

import matplotlib.pyplot as plt

def monk_parser(file_name, input_list, output_list):
    index = 0
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')

        for row in csv_reader:
            value = [0] * 17
            value[int(row[2]) - 1] = 1
            value[3 + int(row[3]) - 1] = 1
            value[6 + int(row[4]) - 1] = 1
            value[8 + int(row[5]) - 1] = 1
            value[11 + int(row[6]) - 1] = 1
            value[15 + int(row[7]) - 1] = 1
            input_list.append(value)
            output_list.append(float(row[1]))
            index = index + 1
        csv_file.close()
        return output_list


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
        np.save(fold_directory_name + "/training_errors_mse", fold['training_curve_errors'])
        np.save(fold_directory_name + "/validation_errors_mse", fold['validation_curve_errors'])
        np.save(fold_directory_name + "/training_errors_mee", fold['validation_curve_errors'])
        np.save(fold_directory_name + "/validation_errors_mee", fold['validation_evaluation_errors'])
        with open(fold_directory_name + '/result.json', 'w') as fp:
            json.dump({'validation_score': fold['validation_score']}, fp)


def generatefolds(X_train, Y_train):
    X_train_splitted, Y_train_splitted = KFoldCrossValidation.split_in_k_fold(X_train, Y_train, 5)

    i = 1
    for fold in X_train_splitted:
        np.save('k-folds/X_%d' % i, fold)
        i += 1

    i = 1
    for fold in Y_train_splitted:
        np.save('k-folds/Y_%d' % i, fold)
        i += 1


if __name__ == '__main__':
    # logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    start_time_GS = datetime.datetime.now().timestamp()

    w_init = RandomNormalInitializer()

    gds = CascadeCorrelationTuningSpecs(
        input_size=20,
        output_layer_list=[Layer(2, Identity, w_init)],
        learning_rate_list=[0.5, 0.25, 0.1],
        momentum_list=[0.1, 0.2, 0.3, 0.45, 0.6],
        regularization_correlation_list=[0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001],
        regularization_pseudo_inverse_list=[0.1, 0.07, 0.04],
        max_nodes_list=[300],
        pool_size_list=[10],
        epochs_list=[10000],
        weights_initializer_list=[w_init],
        activation_function_list=[Sigmoid],
        minimal_correlation_increase_list=[0.001],
        max_fails_increase_list=[50]
    )

    X_folds = []
    Y_folds = []

    for i in range(1, 6):
        X_folds.append(np.load("k-folds/X_%d.npy" % i, allow_pickle=False))
        Y_folds.append(np.load("k-folds/Y_%d.npy" % i, allow_pickle=False))

    cross_validation = KFoldCrossValidation((X_folds, Y_folds), MeanSquaredError, MeanEuclideanError)

    cross_validation.add_early_stopping(
        EarlyStoppingMinimalIncrease(0.0001, 20)
    )
    cross_validation.add_early_stopping(
        EarlyStoppingValidationScore(10)
    )

    gs = GridSearch(gds, cross_validation)

    initial_path = create_timestamp_directory("./grid_search/", "GS-")

    grid_result = gs.run(4, save_result)

    end_time_GS = datetime.datetime.now().timestamp()
    grid_search_duration_in_sec = end_time_GS - start_time_GS
    save_grid_time(grid_search_duration_in_sec)

    sys.exit(0)
