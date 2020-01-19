from typing import List

import numpy as np

from neural_network import NeuralNetwork
from neural_network.learning_algorithm import LearningAlgorithm
from neural_network.learning_observer import ErrorObserver
from neural_network.model_selection import CrossValidation


class KFoldCrossValidation(CrossValidation):
    def __init__(self, k, error_function):
        super().__init__()
        self.__k = k
        self.__error_function = error_function

    def estimates(
            self,
            neural_network: NeuralNetwork,
            learning_algorithm: LearningAlgorithm,
            X_train: np.mat,
            Y_train: np.mat
    ):
        mean = 0
        variance = 0
        errors = []
        for fold in self.__k_folds(X_train, Y_train):
            validation_observer = ErrorObserver(neural_network, fold[1][0], fold[1][1], self.__error_function)
            learning_algorithm.attach(
                validation_observer
            )
            self._on_fold_attempt_start(validation_observer, learning_algorithm)
            learning_algorithm.train(
                neural_network,
                fold[0][0],
                fold[0][1]
            )
            neural_network.pack()
            self._on_fold_attempt_end(validation_observer, learning_algorithm)

            error = min(validation_observer.store)
            errors.append(error)
            mean = mean + error
        mean = mean / len(errors)
        for error in errors:
            variance = variance + np.square(error - mean)
        variance = variance / len(errors)
        self._on_finish(mean, variance)

    def __k_folds(self, X_train: np.mat, Y_train: np.mat) -> List:
        random_indexes = np.arange(len(X_train.T))
        np.random.shuffle(random_indexes)
        X_train_shuffled = X_train[:, random_indexes]
        Y_train_shuffled = Y_train[:, random_indexes]

        X_train_splitted = np.array_split(X_train_shuffled, self.__k, axis=1)
        Y_train_splitted = np.array_split(Y_train_shuffled, self.__k, axis=1)

        result = []
        for i in range(0, self.__k):
            X_val_fold = np.mat(X_train_splitted[i])
            Y_val_fold = np.mat(Y_train_splitted[i])
            X_train_fold = []
            Y_train_fold = []
            for j in range(0, self.__k):
                if j != i:
                    X_train_fold.append(np.mat(X_train_splitted[j]))
                    Y_train_fold.append(np.mat(Y_train_splitted[j]))
            X_train_fold = np.concatenate(X_train_fold, axis=1)
            Y_train_fold = np.concatenate(Y_train_fold, axis=1)
            result.append(
                ((X_train_fold, Y_train_fold), (X_val_fold, Y_val_fold))
            )

        return result
