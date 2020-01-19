from typing import List

import numpy as np

from neural_network import NeuralNetwork
from neural_network.learning_algorithm import LearningAlgorithm
from neural_network.learning_observer import ErrorObserver
from neural_network.model_selection import CrossValidation


class KFoldCrossValidation(CrossValidation):
    def __init__(self, k, error_function):
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
        folds = {}
        fold_counter = 1
        for fold in self.__k_folds(X_train, Y_train):
            initial_weights = [layer.weights.copy() for layer in neural_network.layers]
            validation_observer = ErrorObserver(neural_network, fold[1][0], fold[1][1], self.__error_function)
            training_observer = ErrorObserver(neural_network, fold[0][0], fold[0][1], self.__error_function)
            learning_algorithm.attach(
                validation_observer
            )
            learning_algorithm.attach(
                training_observer
            )
            self._attach_early_stopping(validation_observer, learning_algorithm)
            learning_algorithm.train(
                neural_network,
                fold[0][0],
                fold[0][1]
            )
            neural_network.pack()

            error = min(validation_observer.store)
            errors.append(error)
            mean = mean + error

            folds.update({
                str(fold_counter): {
                    "X_train": fold[0][0],
                    "Y_train": fold[0][1],
                    "X_val": fold[1][0],
                    "Y_val": fold[1][1],
                    "initial_weights": initial_weights,
                    "training_errors": training_observer.store,
                    "validation_errors": validation_observer.store,
                    "validation_score": error
                }
            })
            fold_counter += 1

        mean = mean / len(errors)
        for error in errors:
            variance = variance + np.square(error - mean)
        variance = variance / len(errors)

        folds.update({
            "mean": mean,
            "variance": variance
        })
        return folds

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
