#!/usr/bin/env python3
"""Module de calcul d'intersection bayésienne"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    Calcule l'intersection entre les données observées
        et les probabilités hypothétiques

    Args:
        x: Nombre de patients avec effets secondaires
        n: Nombre total de patients
        P: Tableau 1D de probabilités hypothétiques
        Pr: Tableau 1D des croyances initiales (priors)

    Returns:
        Tableau numpy des intersections pour chaque probabilité

    Exemple:
        >>> intersection(26, 130, [0.1, 0.2], [0.5, 0.5])
        array([1.35665479e-04, 4.35900035e-02])
    """
    # Vérification des erreurs dans l'ordre spécifié
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)) or np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    # Calcul de la vraisemblance (identique à la Tâche 0)
    results = np.zeros_like(P, dtype=np.float64)
    mask = (P > 0) & (P < 1)
    valid_P = P[mask]

    if valid_P.size > 0:
        log_coef = 0.0
        for i in range(1, x + 1):
            log_coef += np.log(n - x + i) - np.log(i)

        log_likelihood = log_coef + x * \
            np.log(valid_P) + (n - x) * np.log(1 - valid_P)
        results[mask] = np.exp(log_likelihood)

    results[P == 0] = 0.0 if x > 0 else 1.0
    results[P == 1] = 0.0 if x < n else 1.0

    # Calcul final de l'intersection
    return results * Pr
