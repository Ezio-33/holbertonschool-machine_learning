#!/usr/bin/env python3
"""
Module définissant un réseau de neurones avec une couche cachée
pour la classification binaire
"""
import numpy as np


class NeuralNetwork:
    """
    Classe définissant un réseau de neurones avec une couche cachée
    et des attributs privés pour la classification binaire
    """

    def __init__(self, nx, nodes):
        """
        Initialise le réseau de neurones

        Args:
            nx (int): nombre de caractéristiques d'entrée
            nodes (int): nombre de nœuds dans la couche cachée

        Raises:
            TypeError: si nx ou nodes n'est pas un entier
            ValueError: si nx ou nodes est inférieur à 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter pour W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter pour b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter pour A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter pour W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter pour b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter pour A2"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calcule la propagation avant du réseau de neurones

        Args:
            X (numpy.ndarray): matrice d'entrée de forme (nx, m)
                nx est le nombre de caractéristiques en entrée du neurone
                m est le nombre d'exemples

        Returns:
            tuple: (A1, A2) les sorties activées des couches
            cachée et de sortie
        """
        # Calcul de la couche cachée
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        # Calcul de la couche de sortie
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2
