#!/usr/bin/env python3
"""Module pour calculer une matrice de corrélation à partir d'une matrice de covariance."""

import numpy as np

def correlation(C):
    """
    Calcule la matrice de corrélation à partir d'une matrice de covariance.

    Args:
        C (numpy.ndarray): Matrice de covariance, forme (d, d).

    Returns:
        numpy.ndarray: Matrice de corrélation, forme (d, d).

    Raises:
        TypeError: Si C n'est pas un numpy.ndarray.
        ValueError: Si C n'est pas une matrice carrée 2D.
    """
    # Vérification du type
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    # Vérification si C est une matrice carrée
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Calcul des écarts-types (racines carrées des variances sur la diagonale)
    std_devs = np.sqrt(np.diag(C))

    # Création d'une matrice contenant tous les produits des écarts-types
    outer_std = np.outer(std_devs, std_devs)

    # Calcul de la matrice de corrélation
    correlation_matrix = C / outer_std

    return correlation_matrix
