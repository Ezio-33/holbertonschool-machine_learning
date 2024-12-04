#!/usr/bin/env python3
"""
Ce module contient une fonction pour concaténer deux matrices 2D.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatène deux matrices 2D selon un axe spécifié.

    Args:
        mat1: Première matrice 2D.
        mat2: Deuxième matrice 2D.
        axis: Axe de concaténation (0 pour vertical, 1 pour horizontal).

    Return:
        Nouvelle matrice résultant de la concaténation.
        None: Si les matrices ne peuvent pas être concaténées.
    """
    # Créer une copie des matrices pour garder l'originales
    result = [row[:] for row in mat1]

    # Concaténation selon l'axe 0
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        result.extend([row[:] for row in mat2])

    # Concaténation selon l'axe 1
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        for i in range(len(mat1)):
            result[i].extend(mat2[i])

    else:
        return None

    return result
