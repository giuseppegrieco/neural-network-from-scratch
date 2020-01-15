import numpy as np


class Loss(object):
    def evaluate(self, predicted_Y, Y):
        pass

    def derivative(self, predicted_Y, Y):
        pass


class MSE(Loss):
    def evaluate(self, predicted_Y, Y):
        return np.mean(
            np.power(predicted_Y - Y, 2)
        )

    def derivative(self, predicted_Y, Y):
        return - (Y - predicted_Y)