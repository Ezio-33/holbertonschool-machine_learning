#!/usr/bin/env python3
"""Implémentation optimisée de K-means avec gestion des clusters vides"""

import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """Effectue le clustering K-means en respectant les contraintes

    Args:
        X (numpy.ndarray): Données de forme (n, d)
        k (int): Nombre de clusters
        iterations (int): Itérations maximum

    Returns:
        tuple: (Centroïdes, Affectations)
    """

    # Validation des entrées
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    n, d = X.shape

    if not isinstance(k, int) or k <= 0 or k >= n:
        return None, None

    # Initialisation des centroïdes
    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    prev_centroids = np.copy(centroids)

    for _ in range(iterations):
        # Calcul des distances (vectorisé)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Mise à jour des centroïdes
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = X[labels == i]

            if cluster_points.size == 0:  # Cluster vide
                # Réutilisation de la fonction d'initialisation
                new_centroids[i] = initialize(X, 1)[0]
            else:
                new_centroids[i] = cluster_points.mean(axis=0)

        # Vérification de convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = np.copy(new_centroids)

    # Dernière assignation
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    return centroids, labels
