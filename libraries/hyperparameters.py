class Hyperparameters:
    def __init__(self,
                 inputNodes,
                 outputLayer,
                 hiddenLayers,
                 learningRate):
        self.inputNodes = inputNodes
        self.outputLayer = outputLayer
        self.hiddenLayers = hiddenLayers
        self.learningRate = learningRate

    def getInputNodes(self):
        return self.inputNodes

    def getOutputLayer(self):
        return self.outputLayer

    def getHiddenLayer(self):
        return self.hiddenLayers

    def getLearningRate(self):
        return self.learningRate

    def setInputNodes(self, inputNodes):
        self.inputNodes = inputNodes

    def setOutputNodes(self, outputLayers):
        self.outputLayer = outputLayers

    def setHiddenLayers(self, hiddenLayers):
        self.hiddenLayers = hiddenLayers

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate