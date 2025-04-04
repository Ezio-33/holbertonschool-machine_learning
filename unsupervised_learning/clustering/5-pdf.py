#!/usr/bin/env python3
"""Calcule la densité de probabilité d'une distribution Gaussienne multivariée"""

import numpy as np

def pdf(X, m, S):
    """Calcule la PDF pour chaque point selon une Gaussienne multivariée
    
    Args:
        X (np.ndarray): Données (n, d)
        m (np.ndarray): Moyenne (d,)
        S (np.ndarray): Matrice de covariance (d, d)
    
    Returns:
        np.ndarray: Probabilités (n,) avec minimum 1e-300
    """
    # Vérification des entrées
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    d = m.shape[0]
    if S.shape != (d, d) or X.shape[1] != d:
        return None
    
    # Calcul du déterminant et de l'inverse de S
    det = np.linalg.det(S)
    inv_S = np.linalg.inv(S)
    
    # Terme de normalisation
    norm = 1 / (np.sqrt((2 * np.pi) ** d * det))
    
    # Différence entre X et m
    diff = X - m
    
    # Calcul de l'exposant (vectorisé)
    exponent = -0.5 * np.sum(diff @ inv_S * diff, axis=1)
    
    # Calcul final de la PDF
    P = norm * np.exp(exponent)
    
    # Application du minimum 1e-300
    P = np.maximum(P, 1e-300)
    
    return P
