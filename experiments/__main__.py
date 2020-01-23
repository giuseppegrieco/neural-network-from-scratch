import csv

import neural_network as nn
import neural_network.utils as utils
import sys
import json
import matplotlib.pyplot as plt

import numpy as np

from neural_network import NeuralNetwork, Layer

def plotgraph(training_errors, validation_errors):
    fig, subplot = plt.subplots(nrows=1, ncols=1)
    subplot.plot(training_errors, '-r', label='Training')
    subplot.plot(validation_errors, '--g', label='Validation')
    subplot.legend()
    plt.show()

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

if __name__ == '__main__':
    X_train = []
    Y_train = []
    X_val  = []
    Y_val = []
    file_parser('monks-3.train', X_train, Y_train)
    file_parser('monks-3.test', X_val, Y_val)

    X_train = np.mat(X_train).T
    print(X_train.shape)
    X_val = np.mat(X_val).T
    print(X_val.shape)
    Y_val = np.mat(Y_val)
    print(Y_val.shape)
    Y_train = np.mat(Y_train)
    print(Y_train.shape)

    mynn = NeuralNetwork(
        17,
        [
            Layer(2, nn.sigmoid),
            Layer(1, nn.sigmoid)
        ]
    )
    tr_errors = []
    vl_errors = []
    print(len(X_train.T))
    for i in range(0, 800):
        tr_errors.append(nn.gradient_descent(
            learning_rate=0.9 / len(X_train.T),
            alpha_momentum=0.8,
            lambda_regularization=0,
            neural_network=mynn,
            input_data=X_train,
            expected_output=Y_train
        ))

        vt = mynn.feed_forward(X_val)
        expected_output = np.mat(Y_val, dtype=np.dtype('d'))

        validation_error = np.matrix.mean(
            np.power(expected_output - vt, 2)
        )

        vl_errors.append(validation_error)
    print(vl_errors[-1])
    plotgraph(tr_errors, vl_errors)

    sys.exit()
    if len(sys.argv) < 2:
        raise Exception("Input Error")

    input_file = sys.argv[1]

    with open('./experiments/' + input_file, 'r') as fp:
        data = json.load(fp)
    is_cup,\
        input_size, \
        training_file, \
        validation_file, \
        epochs, \
        a_eta, \
        a_lambda_reg, \
        a_alpha_momentum, \
        learning_algorithm, \
        a_topology, \
        thread_number,\
        k = utils.read_input(data)


    if is_cup:
        tr_input, tr_output = utils.cup_parser(training_file)
    else:
        tr_input, tr_output = utils.monk_parser(training_file)

    nn.grid_search(
        input_size,
        tr_input,
        tr_output,
        epochs,
        a_eta,
        a_lambda_reg,
        a_alpha_momentum,
        learning_algorithm,
        a_topology,
        thread_number,
        k
    )

    sys.exit()
