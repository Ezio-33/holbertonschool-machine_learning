#!/usr/bin/env python3
"""Module pour la concaténation de matrices NumPy"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatène deux matrices NumPy selon un axe spécifié.

    Args:
        mat1: Première matrice à concaténer.
        mat2: Deuxième matrice à concaténer.
        axis: Axe le long duquel effectuer la concaténation.

    Return:
        Nouvelle matrice concaténé de mat1 et mat2.
    """
    return np.concatenate((mat1, mat2), axis=axis)
