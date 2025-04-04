#!/usr/bin/env python3
"""Implémentation de l'algorithme K-means avec gestion des clusters vides"""

import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """
    Effectue le clustering K-measur les données fournies

    Args:
        X (numpy.ndarray): Jeu de données de forme (n, d)
        k (int): Nombre de clusters souhaité
        iterations (int): Nombre maximum d'itérations

    Returns:
        tuple: (centroïdes finaux, affectations des points)
    """

    # Validation des entrées
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    n, d = X.shape  # Récupère le nombre de points (n) et dimensions (d)

    # Initialisation des centroïdes avec la fonction de la tâche 0
    centroids = initialize(X, k)
    if centroids is None:  # Vérification de l'initialisation
        return None, None

    # Copie des centroïdes pour détection de convergence
    previous_centroids = np.zeros_like(centroids)

    # Boucle principale d'optimisation
    for _ in range(iterations):
        # Étape 1 : Calcul des distances entre points et centroïdes (vectorisé)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # Attribution des points au cluster le plus proche
        clusters = np.argmin(distances, axis=1)

        # Sauvegarde des anciens centroïdes pour comparaison
        previous_centroids[:] = centroids  # Utilisation de slice pour copie

        # Étape 2 : Mise à jour des centroïdes
        for cluster_idx in range(k):
            # Points appartenant au cluster courant
            cluster_points = X[clusters == cluster_idx]

            if cluster_points.size == 0:  # Cas du cluster vide
                # Réinitialisation avec la méthode de la tâche 0
                new_centroid = initialize(X, 1)
                if new_centroid is not None:
                    centroids[cluster_idx] = new_centroid[0]
            else:
                # Calcul du nouveau centroïde comme moyenne des points
                centroids[cluster_idx] = cluster_points.mean(axis=0)

        # Vérification de la convergence (amélioration avec tolérance
        # numérique)
        if np.allclose(centroids, previous_centroids, atol=1e-5, rtol=0):
            break

    # Dernière attribution des clusters après convergence
    final_distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    final_clusters = np.argmin(final_distances, axis=1)

    return centroids, final_clusters
