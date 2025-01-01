#!/usr/bin/env python3
"""
new class: DeepNeuralNetwork, task  16:  DeepNeuralNetwork
"""
import numpy as np


class DeepNeuralNetwork:
    """ define a new class"""

    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.layers = layers
        self.L = len(layers)
        self.weights = dict()
        self.cache = dict()

        for i in range(len(layers)):
            if type(layers[i]) is not int or layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.weights['W1'] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights['W{}'.format(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.weights['b{}'.format(i + 1)] = np.zeros((layers[i], 1))
