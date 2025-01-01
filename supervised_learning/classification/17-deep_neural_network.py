#!/usr/bin/env python3
"""
Module définissant un réseau de neurones profond
pour la classification binaire
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Classe définissant un réseau de neurones profond
    avec des attributs privés pour la classification binaire
    """

    def __init__(self, nx, layers):
        """
        Initialise un réseau de neurones profond

        Args:
        nx (int): nombre de caractéristiques d'entrée
        layers (list): liste contenant le nombre de nœuds pour chaque couche

        Raises:
        TypeError: si nx n'est pas un entier ou si layers n'est pas une liste
        ValueError: si nx ou un élément de layers est inférieur à 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(len(layers)):
            if type(layers[i]) is not int or layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights['W1'] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights[f'W{i + 1}'] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights[f'b{i + 1}'] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter pour L"""
        return self.__L

    @property
    def cache(self):
        """Getter pour cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter pour weights"""
        return self.__weights
