#!/usr/bin/env python3
"""
Module qui calcule les constantes de normalisation d'une matrice
"""
import numpy as np


def normalization_constants(X):
    """
    Calcule la moyenne et l'écart-type de chaque caractéristique d'une matrice.

    Args:
        X: numpy.ndarray de forme (m, nx) à normaliser
            m est le nombre de points de données
            nx est le nombre de caractéristiques

    Returns:
        mean: moyenne de chaque caractéristique
        std: écart-type de chaque caractéristique
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
