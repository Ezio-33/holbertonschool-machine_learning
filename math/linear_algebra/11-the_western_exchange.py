#!/usr/bin/env python3
"""Module pour transposer une matrice NumPy"""

import numpy as np


def np_transpose(matrix):
    """
    Transpose une matrice NumPy.

    Args:
        matrix (numpy.ndarray): La matrice à transposer.

    Returns:
        numpy.ndarray: La matrice transposée.
    """
    return matrix.T
