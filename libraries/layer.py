# NN Layer class
class Layer:
    def __init__(self, nodes, activationFunction):
        self.activationFunction = activationFunction
        self.nodes = [None] * nodes

    def getWeight(self, node):
        return self.nodes[node]

    def setWeight(self, node, value):
        self.nodes[node] = value

    def computes(self, previousLayer):
        pass

    def getActivationFunction(self):
        return self.activationFunction

    def setActivationFunction(self, activationFunction):
        self.activationFunction = activationFunction