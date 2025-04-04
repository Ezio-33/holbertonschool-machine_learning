#!/usr/bin/env python3
"""Étape de maximisation pour l'algorithme EM des GMM"""

import numpy as np


def maximization(X, g):
    """Met à jour les paramètres du modèle GMM

    Args:
        X: ndarray (n, d) - Données
        g: ndarray (k, n) - Probabilités postérieures

    Returns:
        tuple: (pi, m, S) mis à jour
    """
    # Validation des entrées
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    k, n = g.shape
    d = X.shape[1]

    # Calcul des nouveaux poids (pi)
    pi = np.sum(g, axis=1) / n

    # Calcul des nouvelles moyennes (m)
    m = (g @ X) / np.sum(g, axis=1).reshape(k, 1)

    # Calcul des nouvelles covariances (S)
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        weighted = g[i].reshape(n, 1, 1) * np.einsum('ij,ik->ijk', diff, diff)
        S[i] = np.sum(weighted, axis=0) / np.sum(g[i]) + 1e-5 * np.eye(d)

    return pi, m, S
