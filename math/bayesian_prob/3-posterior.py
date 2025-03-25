#!/usr/bin/env python3
"""Module de calcul de probabilité postérieure bayésienne"""
import numpy as np


def posterior(x, n, P, Pr):
    """
    Calcule la probabilité postérieure pour chaque hypothèse

    Args:
        x: Nombre de patients avec effets secondaires
        n: Nombre total de patients
        P: Tableau des probabilités hypothétiques
        Pr: Tableau des croyances initiales (priors)

    Returns:
        numpy.ndarray: Probabilités postérieures
    """
    # Vérification des erreurs (identique aux tâches précédentes)
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
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calcul de la vraisemblance
    likelihoods = np.zeros_like(P)
    mask = (P > 0) & (P < 1)
    valid_P = P[mask]

    if valid_P.size > 0:
        log_coef = 0.0
        for i in range(1, x + 1):
            log_coef += np.log(n - x + i) - np.log(i)

        likelihoods[mask] = np.exp(
            log_coef
            + x * np.log(valid_P)
            + (n - x) * np.log(1 - valid_P)
        )

    likelihoods[P == 0] = 0.0 if x > 0 else 1.0
    likelihoods[P == 1] = 0.0 if x < n else 1.0

    # Calcul de l'intersection (Vraisemblance × Prior)
    intersection = likelihoods * Pr

    # Calcul de la marginale (somme des intersections)
    marginal = np.sum(intersection)

    # Calcul du postérieur (normalisation)
    return intersection / marginal
