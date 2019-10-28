import numpy as np

"""
  The follow functions provide an implementation for activation 
  function of a node.
  
  see https://en.wikipedia.org/wiki/Activation_function
"""


def sigmoid(x):
    """
    Implementation of sigmoid function

    Args:
        x: the point where compute the function
    """
    return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
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


def tanhDerivative(x):
    """
    Implementation of tangent's derivative

    Args:
        x: the point where compute the derivative
    """
    return 1 - np.power(tanhDerivative(x), 2)


def relu(x):
    """
    Implementation of ReLU function

    Args:
        x: the point where compute the function
    """
    return max(0, x)


def reluDerivative(x):
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


def softplusDerivative(x):
    """
    Implementation of SoftPlus's derivative

    Args:
        x: the point where compute the derivative
    """
    return 1 / (1 + np.exp(-x))
