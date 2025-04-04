#!/usr/bin/env python3
"""Implémentation de K-means optimisée avec gestion
précise des clusters vides"""

import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """Effectue le clustering K-means en respectant toutes les contraintes

    Args:
        X (numpy.ndarray): Données de forme (n, d)
        k (int): Nombre de clusters
        iterations (int): Nombre maximal d'itérations

    Returns:
        tuple: (centroïdes finaux, affectations des points)
    """

    # Validation des entrées
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None

    n, d = X.shape
    min_vals = np.min(X, axis=0)  # Précalcul pour optimisation
    max_vals = np.max(X, axis=0)

    # Initialisation des centroïdes
    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    # Copie pour détection de convergence
    centroids_prev = np.zeros_like(centroids)

    for _ in range(iterations):
        # Calcul des distances vectorisé
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        # Mise à jour des centroïdes
        np.copyto(centroids_prev, centroids)
        for i in range(k):
            cluster_points = X[clss == i]

            if cluster_points.size == 0:  # Cluster vide
                centroids[i] = np.random.uniform(min_vals, max_vals, size=d)
            else:
                centroids[i] = cluster_points.mean(axis=0)

        # Vérification de convergence
        if np.allclose(centroids, centroids_prev, atol=1e-5):
            break

    # Recalcul final des affectations
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    clss = np.argmin(distances, axis=1)

    return centroids, clss
