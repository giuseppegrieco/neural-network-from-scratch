import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    pass

def tanhDerivative(x):
    pass

def relu(x):
    pass

def reluDerivative(x):
    pass

def softmax(x):
    pass

def softmaxDerivative(x):
    pass