#!/usr/bin/env python3
"""
Module de réduction de dimensionnalité par PCA avec SVD
"""

import numpy as np

def pca(X, ndim):
    """
    Effectue une réduction PCA en spécifiant le nombre de dimensions

    Args:
        X: Matrice de données (n_samples, n_features)
        ndim: Nombre de dimensions désiré en sortie

    Returns:
        T: Matrice transformée (n_samples, ndim)
    """
    # Centrage des données
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Décomposition SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Sélection des composantes principales
    T = np.dot(X_centered, Vt.T[:, :ndim])

    return T
