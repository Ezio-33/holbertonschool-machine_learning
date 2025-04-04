#!/usr/bin/env python3
"""Implémentation optimisée de l'algorithme K-means avec gestion des clusters vides"""

import numpy as np

def kmeans(X, k, iterations=1000):
    """Effectue le clustering K-means en respectant les contraintes du projet
    
    Args:
        X (numpy.ndarray): Jeu de données de forme (n, d)
        k (int): Nombre de clusters désiré
        iterations (int): Nombre maximal d'itérations
        
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

    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    
    # Initialisation aléatoire des centroïdes
    centroids = np.random.uniform(
        low=min_vals,
        high=max_vals,
        size=(k, d)
    )
    
    # Copie pour détection de convergence
    old_centroids = np.empty_like(centroids)
    
    for _ in range(iterations):
        # Calcul des distances (optimisé)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)
        
        # Mise à jour des centroïdes
        np.copyto(old_centroids, centroids)
        for i in range(k):
            cluster_points = X[clss == i]
            
            if cluster_points.size == 0:  # Cluster vide
                centroids[i] = np.random.uniform(min_vals, max_vals, size=(d,))
            else:
                centroids[i] = cluster_points.mean(axis=0)
                
        # Vérification de convergence
        if np.allclose(centroids, old_centroids):
            break
    
    return centroids, clss
