#!/usr/bin/env python3
"""
Module contenant la fonction de conversion en encodage one-hot
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Convertit un vecteur d'étiquettes en matrice one-hot

    Args:
        Y (numpy.ndarray): vecteur d'étiquettes de forme (m,)
            où m est le nombre d'exemples
        classes (int): nombre maximum de classes trouvées dans Y

    Returns:
        numpy.ndarray: Matrice one-hot de forme (classes, m),
        None en cas d'erreur
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None

    try:
        # Crée une matrice one-hot en utilisant np.eye
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception:
        return None
