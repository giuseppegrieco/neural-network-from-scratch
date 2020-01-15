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
from .gradient_descent import gradient_descent
from .cascade_correlation import cascade_correlation
from .utils import *
from .grid_search import grid_search

# Import able to access:
__all__ = [
    'Layer',
    'NeuralNetwork',
    'gradient_descent',
    'cascade_correlation',
    'identity',
    'tanh',
    'sigmoid',
    'convert_in_numpy',
    'save_graph',
    'utils',
    'grid_search'
]
