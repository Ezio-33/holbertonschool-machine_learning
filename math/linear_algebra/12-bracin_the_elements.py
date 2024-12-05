#!/usr/bin/env python3
"""Module pour effectuer des opérations élément
par élément sur deux matrices NumPy"""


def np_elementwise(mat1, mat2):
    """
    Effectue des opérations élément par élément sur deux matrices NumPy.

    Args:
        mat1: Première matrice
        mat2: Deuxième matrice

    Returns:
        tuple: Contient les résultats de l'addition, soustraction,
               multiplication et division élément par élément
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
