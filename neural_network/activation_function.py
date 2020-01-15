import numpy as np


class Identity(object):
    def f(self, x):
        return x

    def d(self, x):
        return np.ones(
            x.shape
        )


class Sigmoid(object):
    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def d(self, x):
        f_x = self.f(x)
        return np.multiply(f_x, (1 - f_x))
