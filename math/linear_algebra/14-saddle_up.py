#!/usr/bin/env python3
"""Module pour effectuer la multiplication de matrices avec NumPy"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Effectue la multiplication de deux matrices NumPy.

    Args:
        mat1 (numpy.ndarray): Première matrice
        mat2 (numpy.ndarray): Deuxième matrice

    Returns:
        numpy.ndarray: Résultat de la multiplication des matrices
    """
    return np.matmul(mat1, mat2)
