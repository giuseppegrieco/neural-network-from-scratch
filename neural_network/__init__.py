# -*- coding: utf-8 -*-
"""
This module contains all classes and functions required to build an
Artificial Neural Network.

Style Guide:
http://google.github.io/styleguide/pyguide.html
"""
from .neural_network import NeuralNetwork
from .layer import Layer
from .activation_function import *
from .learning_algorithm import *

# Import able to access:
__all__ = [
    'Layer',
    'LearningAlgorithm',
    'GradientDescent',
    'NeuralNetwork',
    'ActivationFunction',
    'Tanh',
    'Sigmoid',
]
