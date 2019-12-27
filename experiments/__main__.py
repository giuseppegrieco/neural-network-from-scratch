from concurrent.futures.process import ProcessPoolExecutor

from tornado import concurrent

import neural_network as nn
import matplotlib.pyplot as plt
import csv
import numpy as np
import sys
import random
import string
import os


def run(training, validation, input_size, lambda_reg, alpha_momentum, topology, epochs):
    my_nn = nn.NeuralNetwork(
        input_size=input_size,
        topology=topology,
        learning_algorithm=nn.GradientDescent(
            learning_rate=eta,
            lambda_regularization=lambda_reg,
            alpha_momentum=alpha_momentum
        )
    )

    def computes_accuracy(targets, expected_output):
        res = []
        for i in np.nditer(targets):
            if i > 0.5:
                res.append(1.0)
            else:
                res.append(0.0)

        all_wrong_output = np.matrix.sum(abs(np.mat(res) - expected_output))
        m, n = vt.shape
        return 1.0 - (all_wrong_output / n)

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
        current_verror = 0
        current_terror = 0
        # vect = list(range(len(tr)))
        # random.shuffle(vect)

        tr_errors.append(my_nn.train(tr_input, tr_output))

        vt = my_nn.feed_forward(ts_input)
        expected_output = np.mat(ts_output)

        v_errors.append(
            np.matrix.sum(np.power(expected_output - vt, 2)) * 1 / len(expected_output.T)
        )
        # Calcolo accuracy
        # v_accuracy.append(computes_accuracy(vt,expected_output))

    print(computes_accuracy(vt, expected_output))


    tr_errors.pop(0)
    v_errors.pop(0)

    fig, subplot = plt.subplots(nrows=1, ncols=1)
    subplot.plot(tr_errors, '-b', label='Training')
    subplot.plot(v_errors, '--r', label='Validation')
    subplot.legend()

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    text_str = '\n'.join((
        r'hidden layers=(2, 3, 5)',
        r'eta=%.2f' % (eta,),
        r'lambda=%.2f' % (lambda_reg,),
        r'alpha=%.2f' % (alpha_momentum,)))
    subplot.text(0.695, 0.8, text_str, transform=subplot.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

    all_char = string.ascii_letters + string.digits
    graph_name = "".join(random.choice(all_char) for x in range(random.randint(8, 12)))
    my_path = os.path.dirname(__file__)
    fig.savefig(os.path.dirname(my_path) + "/charts/" + graph_name)

    return


# Read dataset
def file_parser(file_name, input_list, output_list):
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


tr_input = []
tr_output = []
ts_input = []
ts_output = []

file_parser('monks-2.train', tr_input, tr_output)
file_parser('monks-2.test', ts_input, ts_output)

# Hyperparameters range
a_eta = [0.000001]
a_lambda_reg = [0.1]
a_alpha_momentum = [0.6]
a_topology = [[
    nn.Layer(nodes=2, activation_function=nn.Sigmoid()),
    nn.Layer(nodes=1, activation_function=nn.Identity())
]]
input_size = 17
training = (tr_input, tr_output)
validation = (ts_input, ts_output)
epochs = 1000

thread_executor = ProcessPoolExecutor(2)


futures = []
# Generate all possible combinations
for eta in a_eta:
    for lambda_reg in a_lambda_reg:
        for alpha_momentum in a_alpha_momentum:
            for topology in a_topology:
                futures.append(
                    thread_executor.submit(
                        run, training, validation, input_size, lambda_reg, alpha_momentum, topology, epochs
                    )
                )

concurrent.futures.wait(futures)
thread_executor.shutdown()
sys.exit()









