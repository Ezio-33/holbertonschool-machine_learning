#!/usr/bin/env python3
"""
Module définissant un réseau de neurones avec une couche cachée
pour la classification binaire
"""
import numpy as np


class NeuralNetwork:
    """
    Classe implémentant un réseau de neurones avec une couche cachée
    """

    def __init__(self, nx, nodes):
        """
        Initialise le réseau de neurones

        Args:
            nx (int): nombre de caractéristiques d'entrée
            nodes (int): nombre de noeuds dans la couche cachée
        """
        # Vérification des paramètres
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialisation des poids et biais de la couche cachée
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Initialisation des poids et biais de la couche de sortie
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter pour les poids de la couche cachée"""
        return self.__W1

    @property
    def b1(self):
        """Getter pour le biais de la couche cachée"""
        return self.__b1

    @property
    def A1(self):
        """Getter pour la sortie activée de la couche cachée"""
        return self.__A1

    @property
    def W2(self):
        """Getter pour les poids de la couche de sortie"""
        return self.__W2

    @property
    def b2(self):
        """Getter pour le biais de la couche de sortie"""
        return self.__b2

    @property
    def A2(self):
        """Getter pour la sortie activée finale"""
        return self.__A2
