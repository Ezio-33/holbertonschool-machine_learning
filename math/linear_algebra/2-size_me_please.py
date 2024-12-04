#!/usr/bin/env python3

def matrix_shape(matrix):
    """
    Calcule la forme d'une matrice.

    Args:
        matrix (list): Une liste de listes représentant la matrice.

    Return:
        list: Une liste d'entiers représentant les dimensions de la matrice.
    """
    dimensions = []
    current = matrix
    while isinstance(current, list):
        dimensions.append(len(current))
        if len(current) > 0:
            current = current[0]
        else:
            break
    return dimensions
