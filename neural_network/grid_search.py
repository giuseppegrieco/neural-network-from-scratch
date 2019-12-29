"""
  The follow functions provide an implementation of grid-search with k-fold cross-validation.

  see https://en.wikipedia.org/wiki/Activation_function
"""
from concurrent.futures.process import ProcessPoolExecutor
from tornado import concurrent
import neural_network as nn
import neural_network.utils as utils
import numpy as np
import sys
import datetime
import os
import json


def grid_search(input_size,
                tr_input,
                tr_output,
                epochs,
                a_eta,
                a_lambda_reg,
                a_alpha_momentum,
                learning_algorithm,
                a_topology,
                thread_number,
                k):
    folds = utils.split_k_fold(tr_input, tr_output, k)

    training = (tr_input, tr_output)
    thread_executor = ProcessPoolExecutor(thread_number)

    start_time_GS = datetime.datetime.now().timestamp()
    directory_name_GS = utils.create_timestamp_directory("./grid_search/", prefix="GS-")
    directory_name_GS = directory_name_GS + "/"
    futures = []

    # Generate all possible combinations
    for topology in a_topology:  # TODO: forse è meglio mettere come primo for la topologia
        for lambda_reg in a_lambda_reg:
            for alpha_momentum in a_alpha_momentum:
                for eta in a_eta:
                    __run(folds, input_size, lambda_reg, alpha_momentum, topology, epochs, eta,directory_name_GS)

    concurrent.futures.wait(futures)
    thread_executor.shutdown()

    end_time_GS = datetime.datetime.now().timestamp()
    grid_search_duration_in_sec = end_time_GS - start_time_GS
    data = {'duration_GS': grid_search_duration_in_sec}
    with open("./grid_search/" + directory_name_GS + 'data.json', 'w') as fp:
        json.dump(data, fp)


def __run(folds, input_size, lambda_reg, alpha_momentum, topology, epochs, eta, initial_path):
    start_time = datetime.datetime.now().timestamp()
    directory_name = utils.create_timestamp_directory("./grid_search/" + initial_path, "")
    directory_name = initial_path + directory_name + "/"
    my_nn = nn.NeuralNetwork(
        input_size=input_size,
        topology=topology,
        learning_algorithm=nn.GradientDescent(
            learning_rate=eta,
            lambda_regularization=lambda_reg,
            alpha_momentum=alpha_momentum
        )
    )
    average = 0
    final_errors = []

    for folds_index in range(1, len(folds) + 1):

        fold_directory_name = "./grid_search/" + directory_name + "fold-" + str(folds_index) + "/"
        try:
            os.mkdir(fold_directory_name)
        except FileExistsError as e:
            raise e
        tr_in, tr_out, vn_in, vn_out = utils.retrieves_fold_k(
            folds,
            folds_index
        )
        initial_weights = my_nn.get_all_weights()

        tr_errors = []
        v_errors = []

        tr_input = np.mat(data=tr_in, dtype=np.dtype('d')).T
        vn_input = np.mat(data=vn_in, dtype=np.dtype('d')).T
        tr_output = np.mat(data=tr_out, dtype=np.dtype('d')).T
        vn_output = np.mat(data=vn_out, dtype=np.dtype('d')).T

        min_error = sys.float_info.max
        counter = 20
        for epoch in range(1, epochs + 1):
            error = my_nn.train(tr_input, tr_output)
            tr_errors.append(error)

            vt = my_nn.feed_forward(vn_input)
            expected_output = np.mat(vn_output, dtype=np.dtype('d'))

            error = np.matrix.sum(
                np.power(expected_output - vt, 2)
            ) * 1 / len(expected_output.T)

            # Early stopping
            error, min_error, counter, epoch, result = utils.early_stopping(
                error,
                min_error,
                counter,
                epoch
            )

            if result:
                break

            v_errors.append(error)

        tr_errors.pop(0)
        v_errors.pop(0)

        average = average + v_errors[len(v_errors) - 1]
        final_errors.append(v_errors[len(v_errors) - 1])

        final_weights = my_nn.get_all_weights()

        end_time = datetime.datetime.now().timestamp()

        duration_in_sec = end_time - start_time
        utils.save_data(
            directory_name,
            tr_errors,
            v_errors,
            final_weights,
            initial_weights,
            eta,
            lambda_reg,
            alpha_momentum,
            my_nn,
            folds_index)

        my_nn.init_weights()

    variance = 0
    average = average / len(folds)
    for error in final_errors:
        variance = variance + np.square(error - average)

    variance = variance / len(folds)


    utils.create_output_json(eta, lambda_reg, alpha_momentum, epochs, duration_in_sec, my_nn.get_topology(),
                             "./grid_search/" + directory_name,average,variance)

    return average, variance
