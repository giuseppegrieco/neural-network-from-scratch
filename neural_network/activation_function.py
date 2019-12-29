"""
The follow functions provide an implementation for activation function of a node.

see https://en.wikipedia.org/wiki/Activation_function
"""
import numpy as np


def identity(x, derivative=False):
    """"
    Identity function

    Args:
        x: the point where compute the function
        derivative: true if you want to output the derivative

    Returns:
        the value computed by function or its derivative
    """
    if derivative:
        return np.ones(
            x.shape,
            dtype=np.dtype('d')
        )
    else:
        return x


def tanh(x, derivative=False):
    """"
    Tangent function

    Args:
        x: the point where compute the function
        derivative: true if you want to output the derivative

    Returns:
        the value computed by function or its derivative
    """
    if derivative:
        1 - np.power(tanh(x, False), 2)
    else:
        return np.tanh(x)


def sigmoid(x, derivative=False):
    """"
    Sigmoid function

    Args:
        x: the point where compute the function
        derivative: true if you want to output the derivative

    Returns:
        the value computed by function or its derivative
    """
    if derivative:
        val_x = sigmoid(x, False)
        return np.multiply(val_x, 1 - val_x)
    else:
        return 1 / (1 + np.exp(-x))
