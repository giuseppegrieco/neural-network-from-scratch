import numpy as np


class Layer:
    def __init__(self, previous_nodes, nodes, activation_function):
        self.activation_function = activation_function
        self.weights = np.random.rand(nodes, previous_nodes)

    def getNodeWeights(self, node):
        return self.weights[node]

    def setNodeWeights(self, node, weights):
        self.weights[node] = weights

    def getWeights(self):
        return self.weights

    def computes(self, inputs):
        return self.activation_function(self.weights.dot(inputs))

    def getActivationFunction(self):
        return self.activation_function

    def setActivationFunction(self, activation_function):
        self.activation_function = activation_function
