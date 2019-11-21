import numpy as np
import matplotlib.pyplot as plt
import random
import string


def convert_in_numpy(target):
    if isinstance(target, list) and not isinstance(target[0], list):
        target = [1] + target
        target = np.array(target, dtype=float)
        return target.reshape((len(target), 1))
    else:
        target = np.array(target)
        n, m = target.shape
        bias = np.ones((1, m))
        return np.vstack((bias, target))


def save_graph(terrors, verrors, eta, lambda_reg, alpha_momentum):
    fig, subplot = plt.subplots(nrows=1, ncols=1)
    subplot.plot(terrors, '-b', label='Training')
    subplot.plot(verrors, '--r', label='Validation')
    subplot.legend()

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    text_str = '\n'.join((
        r'hidden layers=(2, 3, 5)',
        r'eta=%.2f' % (eta,),
        r'lambda=%.6f' % (lambda_reg,),
        r'alpha=%.2f' % (alpha_momentum,)))
    subplot.text(0.695, 0.8, text_str, transform=subplot.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

    all_char = string.ascii_letters + string.digits
    graph_name = "".join(random.choice(all_char) for x in range(random.randint(8, 12)))
    plt.show()
    fig.savefig(graph_name)