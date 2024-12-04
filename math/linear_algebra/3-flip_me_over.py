#!/usr/bin/env python3
"""
def matrix_transpose(matrix) : renvoie la fonction
transposée d'une matrice 2D.
"""


def matrix_transpose(matrix):
    """
    Transpose une matrice 2D.

    Args:
        matrix: La matrice 2D à transposer.

    Return:
        Une nouvelle matrice qui est la transposée de la matrice d'entrée.
    """
    # Obtenir les dimensions de la matrice
    rows = len(matrix)
    cols = len(matrix[0])

    # Créer une nouvelle matrice avec dimensions inversées
    transposed = [[matrix[j][i] for j in range(rows)] for i in range(cols)]

    return transposed
