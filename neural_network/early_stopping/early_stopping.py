from abc import ABC

from neural_network.learning_observer import ErrorObserver, LearningObserver


class EarlyStopping(LearningObserver, ABC):
    def __init__(
            self
    ):
        self.error_observer = None

    @property
    def error_observer(self) -> ErrorObserver:
        return self._error_observer

    @error_observer.setter
    def error_observer(self, error_observer: ErrorObserver):
        self._error_observer = error_observer
