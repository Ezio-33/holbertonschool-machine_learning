#!/usr/bin/env python3
"""Implémentation de K-means optimisée avec gestion
précise des clusters vides"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Calculer le centroïde par l'algorithme K mean
    retourner les K centroïdes et le clss
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    n, d = X.shape

    # initialiser des k-centroïdes aléatoires
    centroid = np.random.uniform(low=np.min(X, axis=0),
                                 high=np.max(X, axis=0), size=(k, d))

    for i in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroid, axis=2)
        clss = np.argmin(distances, axis=1)

        new_centroid = np.copy(centroid)

        for j in range(k):
            # nouveau centroïde
            if len(np.where(clss == j)[0]) == 0:
                centroid[j] = np.random.uniform(np.min(X, axis=0),
                                                np.max(X, axis=0), d)

            else:
                centroid[j] = np.mean(X[np.where(clss == j)], axis=0)
        # si le centroïde ne change pas, interrompre
        if np.array_equal(new_centroid, centroid):

            break

    distances = np.linalg.norm(X[:, np.newaxis] - centroid, axis=2)
    clss = np.argmin(distances, axis=1)

    return centroid, clss