from concurrent.futures.process import ProcessPoolExecutor

from tornado import concurrent

import neural_network as nn
import neural_network.utils as utils
import numpy as np
import sys
import datetime
import os
import json


def run(training, validation, input_size, lambda_reg, alpha_momentum, topology, epochs, eta):
    def create_timestamp_directory():
        directory_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f")
        try:
            os.mkdir("./grid_search/" + directory_name)
        except FileExistsError as e:
            print(e)
        return directory_name

    directory_name = create_timestamp_directory()

    start_time = datetime.datetime.now().timestamp()
    my_nn = nn.NeuralNetwork(
        input_size=input_size,
        topology=topology,
        learning_algorithm=nn.GradientDescent(
            learning_rate=eta,
            lambda_regularization=lambda_reg,
            alpha_momentum=alpha_momentum
        )
    )

    initial_weights = my_nn.get_all_weights()

    # wrong way !!!!
    tr_errors = []
    v_errors = []
    v_accuracy = []
    vt = 0
    expected_output = 0

    tr_input = np.mat(training[0]).T.tolist()
    ts_input = np.mat(validation[0]).T.tolist()
    tr_output = training[1]
    ts_output = validation[1]
    # tr_output = training.second
    # tr_input = [[0, 0, 1, 1],[0, 1, 0, 1]]
    # tr_output = [0, 1, 1, 0]
    for i in range(1, epochs):
        tr_errors.append(my_nn.train(tr_input, tr_output))

        vt = my_nn.feed_forward(ts_input)
        expected_output = np.mat(ts_output)

        v_errors.append(
            np.matrix.sum(np.power(expected_output - vt, 2)) * 1 / len(expected_output.T)
        )

        # Calcolo accuracy
        # v_accuracy.append(computes_accuracy(vt,expected_output))

    print(utils.computes_accuracy(vt, expected_output))

    tr_errors.pop(0)
    v_errors.pop(0)

    final_weights = my_nn.get_all_weights()
    end_time = datetime.datetime.now().timestamp()

    duration_in_sec = end_time - start_time
    utils.save_data(directory_name, tr_errors, v_errors, final_weights, initial_weights, eta, lambda_reg,
                    alpha_momentum, epochs, duration_in_sec, my_nn)
    return


tr_input = []
tr_output = []
ts_input = []
ts_output = []

utils.monk_parser('monks-2.train', tr_input, tr_output)
utils.monk_parser('monks-2.test', ts_input, ts_output)

# Hyperparameters range
a_eta = [0.1]
a_lambda_reg = [0.0]
a_alpha_momentum = [0.5]
a_topology = [[
    nn.Layer(nodes=2, activation_function=nn.Sigmoid()),
    nn.Layer(nodes=1, activation_function=nn.Sigmoid())
]]
input_size = 17
training = (tr_input, tr_output)
validation = (ts_input, ts_output)
epochs = 500

thread_executor = ProcessPoolExecutor(4)

start_time_GS = datetime.datetime.now().timestamp()
futures = []
# Generate all possible combinations
for eta in a_eta:
    for lambda_reg in a_lambda_reg:
        for alpha_momentum in a_alpha_momentum:
            for topology in a_topology:
                futures.append(
                    thread_executor.submit(
                        run, training, validation, input_size, lambda_reg, alpha_momentum, topology, epochs, eta
                    )
                )
end_time_GS = datetime.datetime.now().timestamp()
grid_search_duration_in_sec = end_time_GS - start_time_GS

concurrent.futures.wait(futures)
thread_executor.shutdown()
sys.exit()
