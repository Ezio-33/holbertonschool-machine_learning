#!/usr/bin/env python3
"""Implémentation de l'algorithme K-means"""

import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """Effectue le clustering K-means sur un dataset

    Args:
        X: ndarray de forme (n, d) - données
        k: int - nombre de clusters
        iterations: int - itérations max

    Returns:
        C: ndarray (k, d) - centroïdes finaux
        clss: ndarray (n,) - index des clusters
    """

    # Validation des entrées
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None

    n, d = X.shape

    # Initialisation des centroïdes
    C = initialize(X, k)
    if C is None:
        return None, None

    # Copie pour détection de changement
    C_prev = np.copy(C)

    for _ in range(iterations):
        # Étape 1: Assignation des points aux clusters
        distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=2))
        clss = np.argmin(distances, axis=0)

        # Étape 2: Mise à jour des centroïdes
        for i in range(k):
            points = X[clss == i]
            if points.size == 0:  # Cluster vide
                C[i] = initialize(X, 1)
            else:
                C[i] = points.mean(axis=0)

        # Vérification de convergence
        if np.allclose(C, C_prev):
            break
        C_prev = np.copy(C)

    return C, clss
