from concurrent.futures.process import ProcessPoolExecutor

import numpy as np

from neural_network.hyperparameter_tuning.tuning_specs import TuningSpecs
from neural_network.model_selection.cross_validation import CrossValidation


class GridSearch(object):
    def __init__(self, grid_search_specification: TuningSpecs, cross_validation: CrossValidation):
        self.__grid_search_specification = grid_search_specification
        self.__cross_validation = cross_validation

    def run(self, number_of_process: int, on_result):
        with ProcessPoolExecutor(max_workers=number_of_process) as executor:
            futures = []
            for hyperparameters in self.__grid_search_specification.combinations_of_hyperparameters():
                futures.append((executor.submit(
                    self.__cross_validation.estimates,
                    self.__grid_search_specification.build_neural_network_object(
                        hyperparameters
                    ),
                    self.__grid_search_specification.build_learning_algorithm_object(
                        hyperparameters
                    )
                ), hyperparameters))
            for future in futures:
                on_result({
                    'result': future[0].result(),
                    'hyperparameters': self.__grid_search_specification.combinations_repr(future[1])
                })
