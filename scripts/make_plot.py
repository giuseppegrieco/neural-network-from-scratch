import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def plotgraph(training_errors, validation_errors, eta, alpha, lam, fn, sn):
    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(10)
    ax.plot(training_errors, '-r', label='Training')
    ax.plot(validation_errors, '--g', label='Validation')
    ax.legend(fontsize=13)
    ax.set(xlabel='Epochs',
           ylabel='MSE',
           title='Monk 2')
    ax.grid()

    textstr = '\n'.join((
        r'$\eta=%f$' % (eta,),
        r'$\alpha=%f$' % (alpha,),
        r'$\lambda=%f$' % (lam,),
        r'nodes = (%d, %d)' % (fn, sn),
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(1.03, 0.99, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props)
    plt.tight_layout()

    plt.savefig('plot.svg', format="svg")

    plt.show()



file = sys.argv[1]

for directory in os.listdir(file):
    if directory != '.DS_Store' and directory != "run.json":
        with open(file + directory + '/hyperparameters.json', 'r') as myfile:
            data = myfile.read()

        obj = json.loads(data)
        layers = [[int(s) for s in layer_repr.split() if s.isdigit()] for layer_repr in obj['layers']]
        topology = layers[0]
        lr = float(obj['learning_rate'])
        lambda_reg = float(obj['regularization'])
        momentum = float(obj['momentum'])

        for sub_dir in os.listdir(directory):
            tr_errors = np.load(sub_dir + "training_errors_mse.npy", allow_pickle=False)
            vl_errors = np.load(sub_dir + "validation_errors_mse.npy", allow_pickle=False)
            plotgraph(tr_errors, vl_errors, lr, momentum, lambda_reg, topology, 2)

sys.exit()
