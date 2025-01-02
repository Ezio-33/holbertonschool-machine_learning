#!/usr/bin/env python3
"""
Module définissant un réseau de neurones profond
pour la classification binaire.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Classe définissant un réseau de neurones profond
    pour la classification binaire.

    Attributs privés :
        __L (int): Nombre de couches dans le réseau de neurones.
        __cache (dict): Dictionnaire pour stocker les valeurs intermédiaires.
        __weights (dict): Dictionnaire pour stocker les poids
        et biais du réseau de neurones.
    """

    def __init__(self, nx, layers):
        """
        Initialise un réseau de neurones profond.

        Args:
            nx (int): Nombre de caractéristiques d'entrée.
            layers (list): Liste contenant le nombre de nœuds
            pour chaque couche du réseau.

        Raises:
            TypeError: Si nx n'est pas un entier.
            ValueError: Si nx est inférieur à 1.
            TypeError: Si layers n'est pas une liste de nombres positifs.
        """
        # Vérifications des paramètres
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialisation des poids et biais
        for i in range(self.__L):
            layer_num = i + 1
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights[f'W{layer_num}'] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                )
            else:
                self.__weights[f'W{layer_num}'] = (np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1]))
            self.__weights[f'b{layer_num}'] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Retourne le nombre de couches du réseau de neurones."""
        return self.__L

    @property
    def cache(self):
        """Retourne le dictionnaire de cache du réseau de neurones."""
        return self.__cache

    @property
    def weights(self):
        """Retourne le dictionnaire des poids et biais
        du réseau de neurones."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calcule la propagation avant du réseau de neurones

        Args:
                X (numpy.ndarray): données d'entrée de forme (nx, m)
                        nx est le nombre de caractéristiques
                        m est le nombre d'exemples

        Returns:
                tuple: (sortie du réseau, cache)
        """
        self.__cache['A0'] = X

        for index_couche in range(1, self.__L + 1):
            W = self.__weights[f'W{index_couche}']
            b = self.__weights[f'b{index_couche}']
            A_prev = self.__cache[f'A{index_couche - 1}']

            Z = np.matmul(W, A_prev) + b
            self.__cache[f'A{index_couche}'] = 1 / (1 + np.exp(-Z))

        return self.__cache[f'A{self.__L}'], self.__cache

    def cost(self, Y, A):
        """
        Calcule le coût du modèle avec la régression logistique

        Args:
                Y (numpy.ndarray): étiquettes correctes de forme (1, m)
                A (numpy.ndarray): sortie activée du réseau de forme (1, m)

        Returns:
                float: Le coût
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost
