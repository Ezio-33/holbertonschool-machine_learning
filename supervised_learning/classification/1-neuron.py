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
