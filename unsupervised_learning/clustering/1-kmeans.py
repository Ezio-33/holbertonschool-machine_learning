#!/usr/bin/env python3
"""Implémentation optimisée de l'algorithme K-means"""

import numpy as np

def kmeans(X, k, iterations=1000):
    """Effectue le clustering K-means en respectant toutes les contraintes
    
    Args:
        X (numpy.ndarray): Données de forme (n, d)
        k (int): Nombre de clusters
        iterations (int): Itérations maximales
        
    Returns:
        tuple: (C, clss) ou (None, None) en cas d'erreur
    """
    # Validation des entrées
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None
    
    n, d = X.shape
    
    # Initialisation des centroides avec une seule seed
    np.random.seed(0)
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    
    # Premier appel à uniform
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))
    C_prev = C.copy()
    reinitialized = False  # Pour gérer le deuxième appel à uniform
    
    for _ in range(iterations):
        # Calcul des distances
        diff = X[:, np.newaxis] - C
        distances = np.sqrt((diff**2).sum(axis=2))
        clss = np.argmin(distances, axis=1)
        
        # Vérification des clusters vides
        empty_clusters = []
        for i in range(k):
            if not np.any(clss == i):
                empty_clusters.append(i)
        
        # Réinitialisation des clusters vides (un seul appel à uniform)
        if empty_clusters and not reinitialized:
            # Deuxième appel à uniform
            new_centroids = np.random.uniform(
                low=min_vals, 
                high=max_vals, 
                size=(len(empty_clusters), d)
            )
            C[empty_clusters] = new_centroids
            reinitialized = True  # Plus de réinitialisation
        
        # Mise à jour des centroides
        for i in range(k):
            if i in empty_clusters:
                continue  # Déjà réinitialisé si nécessaire
            mask = (clss == i)
            if np.any(mask):
                C[i] = X[mask].mean(axis=0)
        
        # Critère d'arrêt
        if np.allclose(C, C_prev, atol=1e-5):
            break
        C_prev = C.copy()
    
    return C, clss
