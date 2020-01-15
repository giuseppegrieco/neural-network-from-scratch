import neural_network_v0 as nn
import neural_network_v0.utils as utils
import sys
import json
import numpy as np

from neural_network import NeuralNetwork
from neural_network.learning_algorithm.cascade_correlation import CC
from neural_network.loss import MSE

TS = np.genfromtxt('cup/tr.csv', delimiter=',')

TS = TS[:, 1:]

X_train = TS[:1000, :-2]
Y_train = TS[:1000, -2:]

X_val = TS[1000:, :-2]
Y_val = TS[1000:, -2:]
n, m = X_val.T.shape
bias = np.ones((1, m), dtype=np.dtype('d'))
X_val = np.vstack((X_val.T, bias))

print(X_val.shape)

mynn = NeuralNetwork(20)
CC(
    mynn,
    (X_train.T, Y_train.T),
    None,
    50,
    0.01,
    0.8,
    0.001
)
#value = mynn.feed_forward(X_val.T)
#print(MSE().evaluate(value, Y_val.T))

sys.exit()
if __name__ == '__main__':
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
