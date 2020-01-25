import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def plotgraph(directory, training_errors, validation_errors, eta, alpha, lam, fn, sn):
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

    plt.savefig(directory + 'plot.svg', format="svg")

    plt.show()



file = sys.argv[1]

for directory in os.listdir(file):
    if directory != '.DS_Store' and directory != "result.json":

            with open(file + directory + '/hyperparameters.json', 'r') as myfile:
                data = myfile.read()

            obj = json.loads(data)
            lr = float(obj['learning_rate'])
            reg_pseudo = float(obj['regularization_pseudo_inverse'])
            reg_correlation = float(obj['regularization_correlation'])
            momentum = float(obj['momentum'])

            for sub_dir in os.listdir(file + directory):
                if sub_dir != '.DS_Store' and sub_dir != "result.json" and sub_dir != "hyperparameters.json":
                    tr_errors = np.load(file + directory +'/'+ sub_dir + "/training_errors_mse.npy", allow_pickle=False)
                    vl_errors = np.load(file + directory + '/'+ sub_dir + "/validation_errors_mse.npy", allow_pickle=False)
                    plotgraph(file + directory + '/'+ sub_dir + '/', tr_errors, vl_errors, lr, momentum, reg_pseudo, reg_pseudo, 2)
sys.exit()
