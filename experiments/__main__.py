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

    start_time = datetime.datetime.now().timestamp()
    directory_name = create_timestamp_directory()

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

    tr_errors = []
    v_errors = []
    vt = 0
    expected_output = 0

    tr_input = np.mat(training[0]).T.tolist()
    ts_input = np.mat(validation[0]).T.tolist()
    tr_output = training[1]
    ts_output = validation[1]

    min_error = sys.float_info.max
    counter = 20
    for epoch in range(1, epochs + 1):
        error = my_nn.train(tr_input, tr_output)
        tr_errors.append(error)

        vt = my_nn.feed_forward(ts_input)
        expected_output = np.mat(ts_output)

        error = np.matrix.sum(np.power(expected_output - vt, 2)) * 1 / len(expected_output.T)
        # Early stopping
        error, min_error, counter, epoch, result = utils.early_stopping(error, min_error, counter, epoch)
        if result: break

        v_errors.append(error)

    print(utils.computes_accuracy(vt, expected_output))

    tr_errors.pop(0)
    v_errors.pop(0)

    final_weights = my_nn.get_all_weights()
    end_time = datetime.datetime.now().timestamp()

    duration_in_sec = end_time - start_time
    utils.save_data(directory_name, tr_errors, v_errors, final_weights, initial_weights, eta, lambda_reg,
                    alpha_momentum, epochs, duration_in_sec, my_nn)
    return


if len(sys.argv) < 2:
    raise Exception("Input Error")

input_file = sys.argv[1]

with open('./experiments/' + input_file, 'r') as fp:
    data = json.load(fp)

input_size, training_file, validation_file, epochs, a_eta, a_lambda_reg, a_alpha_momentum, learning_algorithm, a_topology, thread_number = utils.read_input(
    data)

tr_input, tr_output = utils.monk_parser(training_file)
ts_input, ts_output = utils.monk_parser(validation_file)

training = (tr_input, tr_output)
validation = (ts_input, ts_output)
thread_executor = ProcessPoolExecutor(thread_number)

start_time_GS = datetime.datetime.now().timestamp()
futures = []

# Generate all possible combinations
for eta in a_eta:  # TODO: forse è meglio mettere come primo for la topologia
    for lambda_reg in a_lambda_reg:
        for alpha_momentum in a_alpha_momentum:
            for topology in a_topology:
                futures.append(
                    thread_executor.submit(
                        run, training, validation, input_size, lambda_reg, alpha_momentum, topology, epochs, eta
                    )
                )
concurrent.futures.wait(futures)
thread_executor.shutdown()

end_time_GS = datetime.datetime.now().timestamp()
grid_search_duration_in_sec = end_time_GS - start_time_GS  # TODO: ha senso salvarla?

sys.exit()
