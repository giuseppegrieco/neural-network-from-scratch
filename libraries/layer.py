import numpy as np


class Layer:
    def __init__(self, previousNodes, nodes, activationFunction):
        self.activationFunction = activationFunction
        self.weights = np.zeros((nodes, previousNodes))

    def getNodeWeights(self, node):
        return self.weights[node]

    def setNodeWeights(self, node, weights):
        self.weights[node] = weights

    def getWeights(self):
        return self.weights

    def computes(self, inputs):
        return self.activationFunction(self.weights.dot(inputs))

    def getActivationFunction(self):
        return self.activationFunction

    def setActivationFunction(self, activationFunction):
        self.activationFunction = activationFunction
