#!/usr/bin/env python3

def matrix_transpose(matrix):
    # Obtenir les dimensions de la matrice
    rows = len(matrix)
    cols = len(matrix[0])

    # Créer une nouvelle matrice avec dimensions inversées
    transposed = [[matrix[j][i] for j in range(rows)] for i in range(cols)]

    return transposed
