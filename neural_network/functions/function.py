from abc import ABC, abstractmethod


class Function(ABC):
    @staticmethod
    @abstractmethod
    def evaluate(x):
        pass

    @staticmethod
    @abstractmethod
    def derivative(x):
        pass
