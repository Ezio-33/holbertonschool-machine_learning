#!/usr/bin/env python3
"""
Module pour déterminer le caractère défini d'une matrice.
"""
import numpy as np


def definiteness(matrix):
    """
    Calcule le caractère défini d'une matrice.

    Args:
        matrix (numpy.ndarray): Matrice dont le caractère
        défini doit être calculé.

    Returns:
        str: "Positive definite", "Positive semi-definite",
        "Negative semi-definite", "Negative definite", ou "Indefinite"
             selon le caractère de la matrice.
             None si la matrice n'est pas valide
             ou ne correspond à aucune catégorie.

    Raises:
        TypeError: Si matrix n'est pas un numpy.ndarray.
    """
    # Vérification du type
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Vérification si la matrice est vide
    if matrix.size == 0:
        return None

    # Vérification si la matrice est carrée
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Vérification si la matrice est symétrique
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Calcul des valeurs propres
        eigenvalues = np.linalg.eigvals(matrix)

        # Vérification que toutes les valeurs propres sont réelles
        if not np.all(np.isreal(eigenvalues)):
            return None

        # Conversion des valeurs propres en réelles
        eigenvalues = np.real(eigenvalues)

        # Détermination du caractère défini
        if np.all(eigenvalues > 0):
            return "Positive definite"
        elif np.all(eigenvalues >= 0) and np.any(eigenvalues == 0):
            return "Positive semi-definite"
        elif np.all(eigenvalues < 0):
            return "Negative definite"
        elif np.all(eigenvalues <= 0) and np.any(eigenvalues == 0):
            return "Negative semi-definite"
        elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
            return "Indefinite"
        else:
            return None

    except np.linalg.LinAlgError:
        # En cas d'erreur lors du calcul des valeurs propres
        return None
