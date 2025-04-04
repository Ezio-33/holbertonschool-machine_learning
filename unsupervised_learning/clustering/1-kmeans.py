#!/usr/bin/env python3
"""Implémentation optimisée de K-means avec gestion des clusters vides"""

import numpy as np
initialize = __import__('0-initialize').initialize

def kmeans(X, k, iterations=1000):
    """Algorithme K-means avec initialisation robuste et
    gestion de convergence"""
    
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
    
    prev_centroids = np.zeros_like(centroids)
    
    for _ in range(iterations):
        # Calcul des distances vectorisées
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)
        
        # Mise à jour des centroïdes
        for i in range(k):
            mask = clss == i
            if not np.any(mask):
                # Réinitialisation selon la même méthode que initialize()
                centroids[i] = initialize(X, 1)
            else:
                centroids[i] = X[mask].mean(axis=0)
                
        # Vérification de convergence
        if np.allclose(centroids, prev_centroids, atol=1e-6):
            break
        prev_centroids = np.copy(centroids)
    
    return centroids, clss
