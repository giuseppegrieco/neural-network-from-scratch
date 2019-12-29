import neural_network as nn
import neural_network.utils as utils
import sys
import json

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
