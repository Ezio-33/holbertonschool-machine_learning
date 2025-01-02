#!/usr/bin/env python3
"""
Module contenant la fonction de décodage one-hot
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Convertit une matrice one-hot en vecteur d'étiquettes numériques

    Args:
        one_hot (numpy.ndarray): matrice one-hot encodée de forme
            (classes, m) où :
            - classes est le nombre maximum de classes
            - m est le nombre d'exemples

    Returns:
        numpy.ndarray: Vecteur de forme (m,) contenant les étiquettes
        numériques pour chaque exemple, None en cas d'erreur
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    try:
        # Utilise argmax pour trouver l'index du 1 dans chaque colonne
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
