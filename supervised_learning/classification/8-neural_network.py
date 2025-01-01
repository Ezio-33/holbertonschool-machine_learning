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
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # Initialisation des poids et biais de la couche de sortie
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
