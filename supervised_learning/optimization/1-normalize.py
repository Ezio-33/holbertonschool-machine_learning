#!/usr/bin/env python3
"""
Module contenant la fonction de normalisation d'une matrice
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalise (standardise) une matrice de données.

    Args:
        X: numpy.ndarray de forme (d, nx) à normaliser
            d est le nombre de points de données
            nx est le nombre de caractéristiques
        m: numpy.ndarray de forme (nx,) contenant les moyennes
           de toutes les caractéristiques de X
        s: numpy.ndarray de forme (nx,) contenant les écarts-types
           de toutes les caractéristiques de X

    Returns:
        La matrice X normalisée
    """
    return (X - m) / s
