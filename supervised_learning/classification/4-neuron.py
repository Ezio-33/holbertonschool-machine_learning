#!/usr/bin/env python3
"""
Module définissant un neurone effectuant une classification binaire
avec calcul du coût
"""
import numpy as np


class Neuron:
    """
    Classe définissant un neurone unique pour la classification binaire
    avec fonction de coût
    """

    def __init__(self, nx):
        """
        Constructeur de la classe Neuron

        Args:
            nx (int): nombre de caractéristiques d'entrée du neurone

        Raises:
            TypeError: si nx n'est pas un entier
            ValueError: si nx est inférieur à 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter pour récupérer les poids"""
        return self.__W

    @property
    def b(self):
        """Getter pour récupérer le biais"""
        return self.__b

    @property
    def A(self):
        """Getter pour récupérer la sortie activée"""
        return self.__A

    def forward_prop(self, X):
        """
        Calcule la propagation avant du neurone

        Args:
            X (numpy.ndarray): matrice d'entrée de forme (nx, m)
                nx est le nombre de caractéristiques
                m est le nombre d'exemples

        Returns:
            float: La sortie activée du neurone
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calcule le coût du modèle avec la régression logistique

        Args:
            Y (numpy.ndarray): étiquettes correctes de forme (1, m)
            A (numpy.ndarray): sortie activée du neurone de forme (1, m)

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
        Évalue les prédictions du neurone

        Args:
            X (numpy.ndarray): matrice d'entrée de forme (nx, m)
            Y (numpy.ndarray): étiquettes correctes de forme (1, m)

        Returns:
            tuple: (prédictions, coût)
                prédictions est une matrice de forme (1, m)
                coût est un scalaire
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
