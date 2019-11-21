import neural_network as nn
import matplotlib.pyplot as plt
import csv
import numpy as np
import sys
import random
import string
import os

# Hyperparameters
eta = 0.01
lambda_reg = 0.0001
alpha_momentum = 0.1
topology = [
    nn.Layer(nodes=5, activation_function=nn.Sigmoid()),
    nn.Layer(nodes=1, activation_function=nn.Sigmoid())
]

nn = nn.NeuralNetwork(
    input_size=6,
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


def file_parser(file_name, input_list, output_list):
    index = 0
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')

        for row in csv_reader:
            input_list.append([
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7])
            ])
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

# wrong way !!!!
tr_errors = []
v_errors = []
v_accuracy = []
vt = 0
expected_output = 0

tr_input = np.mat(tr_input).T.tolist()
ts_input = np.mat(ts_input).T.tolist()
# tr_input = [[0, 0, 1, 1],[0, 1, 0, 1]]
# tr_output = [0, 1, 1, 0]
for i in range(1, 10000):
    current_verror = 0
    current_terror = 0
    # vect = list(range(len(tr)))
    # random.shuffle(vect)

    tr_errors.append(nn.train(tr_input, tr_output))

    vt = nn.feed_forward(ts_input)
    expected_output = np.mat(ts_output)

    v_errors.append(
        np.matrix.sum(np.power(expected_output - vt, 2)) * 1 / len(expected_output.T)
    )
    # Calcolo accuracy
    # v_accuracy.append(computes_accuracy(vt,expected_output))

print(computes_accuracy(vt, expected_output))

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
plt.show()
sys.exit()
