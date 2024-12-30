#!/usr/bin/env python3
"""
Module contenant la classe Neuron qui définit un neurone
effectuant une classification binaire
"""
import numpy as np


class Neuron:
    """
    Classe définissant un neurone unique pour la classification binaire
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

        self.__W = np.random.randn(1, nx)  # Vecteur poids (1 x nx)
        self.__A = 0
        self.__b = 0

    @property
    def W(self):
        """Getter pour récupérer les poids"""
        return self.__W

    @property
    def A(self):
        """Getter pour récupérer la sortie activée"""
        return self.__A

    @property
    def b(self):
        """Getter pour récupérer le biais"""
        return self.__b

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
        # Calcul de la combinaison linéaire Z = W·X + b
        Z = np.matmul(self.__W, X) + self.__b

        # Fonction d'activation sigmoid: 1/(1 + e^(-z))
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A
