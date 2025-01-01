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
            layers (list): liste du nombre de noeuds pour chaque couche
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

        # Boucle d'initialisation des poids et biais
        for index_couche in range(self.__L):
            couche_size = layers[index_couche]
            input_size = nx if index_couche == 0 else layers[index_couche - 1]

            self.__weights[f'W{index_couche+1}'] = (np.random.randn(
                couche_size, input_size) * np.sqrt(2 / input_size))
            self.__weights[f'b{index_couche+1}'] = np.zeros((couche_size, 1))

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
            A_prev = self.__cache[f'A{index_couche-1}']

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

    def evaluate(self, X, Y):
        """
        Évalue les prédictions du réseau de neurones

        Args:
            X (numpy.ndarray): données d'entrée de forme (nx, m)
                nx est le nombre de caractéristiques
                m est le nombre d'exemples
            Y (numpy.ndarray): étiquettes correctes de forme (1, m)

        Returns:
            tuple: (prédictions, coût)
                prédictions est une matrice de forme (1, m)
                coût est un scalaire
        """
        # Propagation avant pour obtenir les prédictions
        A, _ = self.forward_prop(X)

        # Calcul du coût
        cost = self.cost(Y, A)

        # Conversion des probabilités en prédictions binaires
        predictions = np.where(A >= 0.5, 1, 0)

        return predictions, cost
