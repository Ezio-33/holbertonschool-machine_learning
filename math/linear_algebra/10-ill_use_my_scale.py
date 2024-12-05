#!/usr/bin/env python3
"""Module pour calculer la forme d'un tableau NumPy"""

import numpy as np


def np_shape(matrix):
    """
    Calcule la forme d'un tableau NumPy.

    Args:
        matrix (numpy.ndarray): Le tableau NumPy dont on veut
        connaître la forme.

    Return:
        tuple: Un tuple d'entiers représentant la forme du tableau.
    """
    return matrix.shape
