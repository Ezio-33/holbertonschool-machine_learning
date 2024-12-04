#!/usr/bin/env python3
"""fonction qui ajoute deux matrices par élément"""


def add_matrices2D(mat1, mat2):
    """
    Ajoute deux matrices 2D élément par élément.

    Arguments:
        mat1: La première matrice 2D.
        mat2: La deuxième matrice 2D.

    Return:
        Une nouvelle matrice 2D contenant les sommes élément par élément.
        None: Si les matrices n'ont pas les mêmes dimensions.
    """
    # Vérifier si les matrices ont les mêmes dimensions
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    # Créer une nouvelle matrice pour stocker les résultats
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)

    return result
