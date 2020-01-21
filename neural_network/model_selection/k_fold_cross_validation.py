from typing import List

import numpy as np

from neural_network import NeuralNetwork
from neural_network.learning_algorithm import LearningAlgorithm
from neural_network.learning_observer import ErrorObserver
from neural_network.model_selection import CrossValidation
from neural_network.model_selection.error_evaluation_observer import ErrorEvaluationObserver


class KFoldCrossValidation(CrossValidation):
    def __init__(self, folds, error_function_learning_curve, error_function_evaluation):
        super().__init__()
        self.__k = len(folds[0])
        self.__folds = folds
        self.__error_function_learning_curve = error_function_learning_curve
        self.__error_function_evaluation = error_function_evaluation

    def estimates(
            self,
            neural_network: NeuralNetwork,
            learning_algorithm: LearningAlgorithm
    ):
        mean = 0
        variance = 0
        errors = []
        folds = {}
        fold_counter = 1
        for fold in self.folds_combination():
            initial_weights = [layer.weights.copy() for layer in neural_network.layers]
            validation_observer = ErrorEvaluationObserver(
                fold[1][0],
                fold[1][1],
                self.__error_function_learning_curve,
                self.__error_function_evaluation
            )
            training_observer = ErrorEvaluationObserver(
                fold[0][0],
                fold[0][1],
                self.__error_function_learning_curve,
                self.__error_function_evaluation
            )
            learning_algorithm.attach(
                validation_observer
            )
            learning_algorithm.attach(
                training_observer
            )
            super()._attach_early_stopping(
                validation_observer,
                learning_algorithm
            )
            learning_algorithm.train(
                neural_network,
                fold[0][0],
                fold[0][1]
            )

            error = min(validation_observer.store_evaluate)
            errors.append(error)
            mean = mean + error

            folds.update({
                str(fold_counter): {
                    "initial_weights": initial_weights.copy(),
                    "training_curve_errors": training_observer.store.copy(),
                    "validation_curve_errors": validation_observer.store.copy(),
                    "training_evaluation_errors": training_observer.store_evaluate.copy(),
                    "validation_evaluation_errors": validation_observer.store_evaluate.copy(),
                    'validation_score': error
                }
            })
            fold_counter += 1

            # Prepares for new training (new folds combination)
            neural_network.pack()
            learning_algorithm.detach(validation_observer)
            learning_algorithm.detach(training_observer)
            super()._detach_early_stopping(learning_algorithm)

        mean = mean / len(errors)
        for error in errors:
            variance = variance + np.square(error - mean)
        variance = variance / len(errors)

        folds.update({
            "mean": mean,
            "variance": variance
        })
        return folds

    def folds_combination(self) -> List:
        X_train_splitted, Y_train_splitted = self.__folds
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

    @staticmethod
    def split_in_k_fold(X_train, Y_train, k, random=True):
        if random:
            random_indexes = np.arange(len(X_train.T))
            np.random.shuffle(random_indexes)
            X_train_shuffled = X_train[:, random_indexes]
            Y_train_shuffled = Y_train[:, random_indexes]

            X_train_splitted = np.array_split(X_train_shuffled, k, axis=1)
            Y_train_splitted = np.array_split(Y_train_shuffled, k, axis=1)
        else:
            X_train_splitted = np.array_split(X_train, k, axis=1)
            Y_train_splitted = np.array_split(Y_train, k, axis=1)
        return X_train_splitted, Y_train_splitted
