import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanhDerivative(x):
    return 1 - np.power(tanhDerivative(x), 2)

def relu(x):
    return max(0, x)

def reluDerivative(x):
    return relu(x)

def softplus(x):
    return np.log(1 + np.exp(x))

def softplusDerivative(x):
    return 1 / (1 + np.exp(-x))