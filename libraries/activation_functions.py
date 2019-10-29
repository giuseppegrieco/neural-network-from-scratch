"""
  The follow functions provide an implementation for activation function of a node.

  see https://en.wikipedia.org/wiki/Activation_function
"""
import numpy as np


def sigmoid(x):
    """
    Implementation of sigmoid function

    Args:
        x: the point where compute the function
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Implementation of sigmoid's derivative

    Args:
        x: the point where compute the derivative
    """
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    """
    Implementation of tangent function

    Args:
        x: the point where compute the function
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Implementation of tangent's derivative

    Args:
        x: the point where compute the derivative
    """
    return 1 - np.power(tanh(x), 2)


def relu(x):
    """
    Implementation of ReLU function

    Args:
        x: the point where compute the function
    """
    return max(0, x)


def relu_derivative(x):
    """
    Implementation of ReLU's derivative

    Args:
        x: the point where compute the derivative
    """
    return relu(x)


def softplus(x):
    """
    Implementation of SoftPlus function

    Args:
        x: the point where compute the function
    """
    return np.log(1 + np.exp(x))


def softplus_derivative(x):
    """
    Implementation of SoftPlus's derivative

    Args:
        x: the point where compute the derivative
    """
    return 1 / (1 + np.exp(-x))
