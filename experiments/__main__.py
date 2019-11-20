import neural_network as nn
import matplotlib.pyplot as plt
import csv
import numpy as np
import sys
import random
import string

# Hyperparameters
eta = 0.01
lambda_reg = 0.001
alpha_momentum = 0.1
topology = [
    nn.Layer(nodes=4, activation_function=nn.Sigmoid()),
    nn.Layer(nodes=4, activation_function=nn.Sigmoid()),
    nn.Layer(nodes=4, activation_function=nn.Sigmoid()),

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

file_parser('monks-1.train', tr_input, tr_output)
file_parser('monks-1.test', ts_input, ts_output)

# wrong way !!!!
terrors = []
verrors = []

tr_input = np.mat(tr_input).T.tolist()
ts_input = np.mat(ts_input).T.tolist()
#tr_input = [[0, 0, 1, 1],[0, 1, 0, 1]]
#tr_output = [0, 1, 1, 0]
for i in range(1, 10000):
    current_verror = 0
    current_terror = 0
    # vect = list(range(len(tr)))
    # random.shuffle(vect)

    terrors.append(nn.train(tr_input, tr_output))

    vt = nn.feed_forward(ts_input)
    expected_output = np.mat(ts_output)

    verrors.append(
       np.matrix.sum(np.power(expected_output - vt, 2)) * 1 / len(expected_output.T)
    )

    #for j in range(0, len(ts_input)):
        #current_verror = current_verror + np.power(ts_output[j] - nn.feed_forward(ts_input[j]).item(0), 2)
    #current_verror = 1 / len(ts_input) * current_verror
    #verrors.append(current_verror)

tr_input = np.mat(tr_input)
print(nn.feed_forward(tr_input[:,4].tolist()))
print(nn.feed_forward(tr_input[:,10].tolist()))
fig, subplot = plt.subplots(nrows=1, ncols=1)
subplot.plot(terrors, '-b', label='Training')
subplot.plot(verrors, '--r', label='Validation')
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
plt.show()
fig.savefig(graph_name)

sys.exit()
