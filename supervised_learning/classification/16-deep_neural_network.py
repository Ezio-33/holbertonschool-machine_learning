#!/usr/bin/env python3
"""
Module définissant un réseau de neurones profond
pour la classification binaire
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Classe définissant un réseau de neurones profond
    pour la classification binaire
    """

    def __init__(self, nx, layers):
        """
        Initialise un réseau de neurones profond

        Args:
            nx (int): nombre de caractéristiques d'entrée
            layers (list): liste contenant le nombre de neurones par couche
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        # Attributs publics
        self.L = len(layers)  # Nombre de couches
        self.cache = {}       # Stocke les valeurs intermédiaires
        self.weights = {}     # Stocke les poids et biais

        # Initialisation des poids avec la méthode He et al.
        for index_couche in range(self.L):
            couche_size = layers[index_couche]
            input_size = nx if index_couche == 0 else layers[index_couche - 1]

            # Initialisation des poids
            self.weights[f'W{index_couche+1}'] = np.random.randn(
                couche_size, input_size
            ) * np.sqrt(2 / input_size)

            # Initialisation des biais
            self.weights[f'b{index_couche+1}'] = np.zeros((couche_size, 1))
