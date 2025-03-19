#!/usr/bin/env python3
"""
Module pour le calcul de la matrice des cofacteurs d'une matrice.
"""


def determinant(matrix):
    """
    Calcule le déterminant d'une matrice carrée.

    Args:
        matrix (list): Liste de listes représentant la matrice.

    Returns:
        int ou float: Le déterminant de la matrice.
    """
    # Cas particulier matrice 0x0
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    # Taille de la matrice
    size = len(matrix)

    # Cas 1x1
    if size == 1:
        return matrix[0][0]

    # Cas 2x2
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Cas récursif pour matrices >2x2
    det = 0
    for i in range(size):
        # Création de la sous-matrice (mineur)
        sub_matrix = []
        for row in matrix[1:]:  # Ignorer la première ligne
            sub_row = row[:i] + row[i+1:]  # Ignorer la colonne i
            sub_matrix.append(sub_row)

        # Calcul du terme et ajout au déterminant
        sign = (-1) ** i
        det += sign * matrix[0][i] * determinant(sub_matrix)

    return det


def minor(matrix):
    """
    Calcule la matrice des mineurs d'une matrice carrée.

    Args:
        matrix (list): Liste de listes représentant la matrice.

    Returns:
        list: La matrice des mineurs.
    """
    # Vérification du type
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Vérification de liste vide
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Vérification que tous les éléments sont des listes
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Vérification de matrice vide ou non carrée
    size = len(matrix)
    if size == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if len(row) != size:
            raise ValueError("matrix must be a non-empty square matrix")

    # Cas particulier matrice 1x1
    if size == 1:
        return [[1]]

    # Calcul de la matrice des mineurs
    minor_matrix = []
    for i in range(size):
        minor_row = []
        for j in range(size):
            # Création de la sous-matrice en supprimant la ligne i et colonne j
            sub_matrix = []
            for r in range(size):
                if r != i:  # Ignorer la ligne i
                    sub_row = []
                    for c in range(size):
                        if c != j:  # Ignorer la colonne j
                            sub_row.append(matrix[r][c])
                    sub_matrix.append(sub_row)

            # Calcul du déterminant de la sous-matrice
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix


def cofactor(matrix):
    """
    Calcule la matrice des cofacteurs d'une matrice carrée.

    Args:
        matrix (list): Liste de listes représentant la matrice.

    Returns:
        list: La matrice des cofacteurs.

    Raises:
        TypeError: Si matrix n'est pas une liste de listes.
        ValueError: Si matrix n'est pas carrée ou est vide.
    """
    # Vérification du type
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Vérification de liste vide
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Vérification que tous les éléments sont des listes
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Vérification de matrice vide ou non carrée
    size = len(matrix)
    if size == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if len(row) != size:
            raise ValueError("matrix must be a non-empty square matrix")

    # Cas particulier matrice 1x1
    if size == 1:
        return [[1]]

    # Calcul de la matrice des mineurs
    minor_matrix = minor(matrix)

    # Calcul de la matrice des cofacteurs
    cofactor_matrix = []
    for i in range(size):
        cofactor_row = []
        for j in range(size):
            # Application de la formule (-1)^(i+j) * Mineur(i,j)
            sign = (-1) ** (i + j)
            cofactor_row.append(sign * minor_matrix[i][j])
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


def adjugate(matrix):
    """
    Calcule la matrice adjointe d'une matrice carrée.

    Args:
        matrix (list): Liste de listes représentant la matrice.

    Returns:
        list: La matrice adjointe.

    Raises:
        TypeError: Si matrix n'est pas une liste de listes.
        ValueError: Si matrix n'est pas carrée ou est vide.
    """
    # Vérification du type
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Vérification de liste vide
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Vérification que tous les éléments sont des listes
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Vérification de matrice vide ou non carrée
    size = len(matrix)
    if size == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if len(row) != size:
            raise ValueError("matrix must be a non-empty square matrix")

    # Cas particulier matrice 1x1
    if size == 1:
        return [[1]]

    # Calcul de la matrice des cofacteurs
    cofactor_matrix = cofactor(matrix)

    # Transposition de la matrice des cofacteurs pour obtenir l'adjointe
    adjugate_matrix = []
    for j in range(size):  # Parcourir les colonnes
        adjugate_row = []
        for i in range(size):  # Parcourir les lignes
            adjugate_row.append(cofactor_matrix[i][j])
            # Note: i et j sont inversés ici
        adjugate_matrix.append(adjugate_row)

    return adjugate_matrix
