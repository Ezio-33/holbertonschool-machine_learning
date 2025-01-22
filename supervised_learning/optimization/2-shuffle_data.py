#!/usr/bin/env python3
"""
Module contenant une fonction pour mélanger deux matrices de données
de manière synchronisée
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Mélange deux matrices de données de la même façon.

    Args:
        X: numpy.ndarray de forme (m, nx) à mélanger
            m est le nombre de points de données
            nx est le nombre de caractéristiques
        Y: numpy.ndarray de forme (m, ny) à mélanger
            m est le même nombre de points de données que X
            ny est le nombre de caractéristiques dans Y

    Returns:
        Les matrices X et Y mélangées
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
