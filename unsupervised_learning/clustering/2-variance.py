#!/usr/bin/env python3
"""Calcul de la variance intra-cluster pour K-means"""

import numpy as np

def variance(X, C):
    """Calcule la variance totale intra-cluster
    
    Args:
        X (numpy.ndarray): Dataset de forme (n, d)
        C (numpy.ndarray): Centroïdes de forme (k, d)
        
    Returns:
        float: Variance totale ou None en cas d'erreur
    """
    # Vérification des entrées
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    
    # Calcul des distances carrées entre chaque point et chaque centroïde
    differences = X[:, np.newaxis, :] - C  # Forme (n, k, d)
    distances_carrees = np.sum(differences ** 2, axis=2)  # Forme (n, k)
    
    # Sélection de la distance minimale pour chaque point
    distances_min = np.min(distances_carrees, axis=1)  # Forme (n,)
    
    # Somme totale des distances minimales
    return np.sum(distances_min)
