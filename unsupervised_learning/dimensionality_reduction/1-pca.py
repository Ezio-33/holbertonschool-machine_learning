#!/usr/bin/env python3
"""
Réduction de dimensionnalité par PCA avec SVD
Version optimisée pour stabilité numérique
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
    X_centered = X - np.mean(X, axis=0)
    
    # Décomposition en valeurs singulières
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Gestion du cas où ndim > nombre de features
    ndim = min(ndim, X.shape[1])
    
    # Sélection des composantes principales
    T = U[:, :ndim] * S[:ndim]
    
    return T
