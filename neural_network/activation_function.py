"""
  The follow functions provide an implementation for activation function of a node.

  see https://en.wikipedia.org/wiki/Activation_function
"""
import numpy as np


class ActivationFunction:
    """
    Abstract concept of activation function
    """

    """
    Computes the function.
        
    Args:
        x: the point where compute the function
    """
    def f(self, x):
        pass

    """
    Computes the function's derivative.

    Args:
        x: the point where compute the derivative
    """
    def f_derivative(self, x):
        pass

class Identity(ActivationFunction):

    def f(self, x):
        return x

    def f_derivative(self, x):
        return np.ones(x.shape, dtype=np.dtype('d'))

class Sigmoid(ActivationFunction):

    def f(self, x):
        res = 1 / (1 + np.exp(-x))
        return res

    def f_derivative(self, x):
        return np.multiply(self.f(x), (1 - self.f(x)))


class Tanh(ActivationFunction):

    def f(self, x):
        return np.tanh(x)

    def f_derivative(self, x):
        return 1 - np.power(self.f(x), 2)


class ReLU(ActivationFunction):

    def f(self, x):
        pass

    def f_derivative(self, x):
        pass


class Softmax(ActivationFunction):

    def f(self, x):
        pass

    def f_derivative(self, x):
        pass
