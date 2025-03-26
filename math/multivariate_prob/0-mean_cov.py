#!/usr/bin/env python3
"""Module pour calculer la moyenne et la
covariance d'un ensemble de données."""

import numpy as np


def mean_cov(X):
    """
    Calcule la moyenne et la covariance d'un ensemble de données.

    Args:
        X (numpy.ndarray): Ensemble de données de forme (n, d).

    Returns:
        tuple: (mean, cov)
            mean (numpy.ndarray): Moyenne de forme (1, d).
            cov (numpy.ndarray): Matrice de covariance de forme (d, d).

    Raises:
        TypeError: Si X n'est pas un numpy.ndarray 2D.
        ValueError: Si X contient moins de 2 points de données.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    return mean, cov
