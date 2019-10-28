class NeuralNetwork:
    """
    Artificial Neural Network implementation.

    Attributes:
        hyperparameters (:Hyperparameter): It contains all hyperparameters for the nn.
    """
    def __init__(self, hyperparameters):
        """
        Inits NeuralNetwork with the hyperparameters indicated.

        Attributes:
            hyperparameters It contains all hyperparameters to use for this nn.
        """
        self.hyperparameters = hyperparameters
        self.__initWeights()

    def __initWeights(self):
        """
        Computes the initial weights of the network.

        TODO: write the method
        """
        pass

    def train(self, input, expectedOutput):
        """
        Performs the training phase.

        TODO: write the method
        """
        pass

    def computes(self, input):
        """
        Computes the new input and return the result.

        TODO: write the method
        """
        pass
