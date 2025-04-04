#!/usr/bin/env python3
"""Initialisation des paramètres pour un Gaussian Mixture Model"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans

def initialize(X, k):
    """Prépare les paramètres initiaux pour un GMM
    
    Args:
        X: ndarray de forme (n, d) - données
        k: nombre de clusters
        
    Returns:
        tuple: (pi, m, S) initialisés
    """
    # Validation des entrées
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None, None
    
    n, d = X.shape
    
    # 1. Initialisation des priors (équitable)
    pi = np.ones(k) / k  # [1/k, 1/k, ..., 1/k]
    
    # 2. Initialisation des moyennes avec K-means
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None
    
    # 3. Initialisation des matrices de covariance
    S = np.tile(np.eye(d), (k, 1, 1))  # k matrices identité (d x d)
    
    return pi, m, S
