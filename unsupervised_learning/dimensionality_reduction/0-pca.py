#!/usr/bin/env python3
"""
Module PCA avec préservation de variance
Version validée avec les tests
"""

import numpy as np

def pca(X, var=0.95):
    """
    Réduction PCA avec seuil de variance

    Args:
        X: Matrice des données (n_samples, n_features)
        var: Variance à conserver (0-1)

    Returns:
        W: Matrice de projection (n_features, n_components)
    """
    # Centrage des données
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Calcul de covariance "officiel"
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Décomposition propre optimisée
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Tri décroissant
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Calcul variance cumulée
    total_variance = eigenvalues.sum()
    explained_variance = np.cumsum(eigenvalues) / total_variance

    # Sélection composantes
    n_components = np.argmax(explained_variance >= var) + 1
    W = eigenvectors[:, :n_components]

    return W
