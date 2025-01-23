#!/usr/bin/env python3
"""
Module contenant la fonction de batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalise une couche de réseau de neurones en
    utilisant la batch normalization.

    Args:
        Z: numpy.ndarray de forme (m, n) à normaliser
            m est le nombre de données
            n est le nombre de caractéristiques
        gamma: numpy.ndarray de forme (1, n) contenant les facteurs d'échelle
        beta: numpy.ndarray de forme (1, n) contenant les décalages
        epsilon: petit nombre pour éviter la division par zéro

    Returns:
        La matrice Z normalisée
    """
    # Calcul de la moyenne sur l'axe 0 (pour chaque caractéristique)
    mean = np.mean(Z, axis=0, keepdims=True)
    
    # Calcul de la variance sur l'axe 0
    variance = np.var(Z, axis=0, keepdims=True)
    
    # Normalisation
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    
    # Mise à l'échelle et décalage
    Z_scaled = gamma * Z_norm + beta
    
    return Z_scaled
