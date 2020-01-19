from abc import abstractmethod, ABC

import neural_network.learning_algorithm as learning_algorithm


class LearningObserver(ABC):
    """
    The Learning Observer interface declares the update method, used by Learning Algorithms.
    """

    @abstractmethod
    def update(self, learning_algorithm: learning_algorithm.LearningAlgorithm) -> None:
        pass
