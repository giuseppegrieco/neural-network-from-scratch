from abc import ABC

from neural_network.learning_observer import ErrorObserver, LearningObserver


class EarlyStopping(LearningObserver, ABC):
    def __init__(
            self,
            error_observer: ErrorObserver
    ):
        self._error_observer = error_observer
