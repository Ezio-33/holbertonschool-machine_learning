#!/usr/bin/env python3
"""
Module pour le calcul du déterminant d'une matrice.
"""


def determinant(matrix):
    """
    Calcule le déterminant d'une matrice carrée.

    Args:
        matrix (list): Liste de listes représentant la matrice.

    Returns:
        int or float: Le déterminant de la matrice.

    Raises:
        TypeError: Si matrix n'est pas une liste de listes.
        ValueError: Si matrix n'est pas carrée.
    """
    # Vérification du type (doit être une liste)
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Vérification que c'est une liste de listes
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Cas particulier : liste vide []
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Cas particulier : matrice 0×0 représentée par [[]]
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    # Vérification que la matrice est carrée
    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            raise ValueError("matrix must be a square matrix")

    # Cas simple : matrice 1×1
    if size == 1:
        return matrix[0][0]

    # Cas simple : matrice 2×2
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Cas général : matrice n×n (n>2) par développement selon la première ligne
    det = 0
    for i in range(size):
        # Création de la sous-matrice (mineur)
        minor = [row[:i] + row[i+1:] for row in matrix[1:]]
        # Application de la formule avec alternance de signe correcte
        det += matrix[0][i] * (-1) ** i * determinant(minor)

    return det
