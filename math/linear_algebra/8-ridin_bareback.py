#!/usr/bin/env python3
"""
Ce module contient une fonction pour effectuer la multiplication de matrices.
"""


def mat_mul(mat1, mat2):
    """
    Effectue la multiplication de deux matrices.

    Args:
        mat1 (list of lists): Première matrice 2D.
        mat2 (list of lists): Deuxième matrice 2D.

    Returns:
        list of lists: Nouvelle matrice résultant de la multiplication.
        None: Si les matrices ne peuvent pas être multipliées.
    """
    # Vérifier si les matrices peuvent être multipliées
    if len(mat1[0]) != len(mat2):
        return None

    # Initialiser la matrice résultante
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Effectuer la multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
