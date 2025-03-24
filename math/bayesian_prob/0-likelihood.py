#!/usr/bin/env python3
"""Module de calcul de vraisemblance bayésienne"""
import numpy as np


def likelihood(x, n, P):
    """
    Calcule la vraisemblance binomiale pour des probabilités hypothétiques

    Args:
        x: Nombre de succès observés (26 effets secondaires)
        n: Nombre total d'essais (130 patients)
        P: Tableau numpy de probabilités hypothétiques

    Returns:
        Tableau numpy des vraisemblances pour chaque p dans P
    """
    # Vérifications d'erreurs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n doit être un entier positif")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x doit être un entier positif ou nul")
    if x > n:
        raise ValueError("x ne peut pas être supérieur à n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P doit être un tableau numpy 1D")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("Les valeurs de P doivent être entre 0 et 1")

    # Initialisation du résultat
    results = np.zeros_like(P, dtype=np.float64)

    # Gestion des cas particuliers
    mask = (P > 0) & (P < 1)
    valid_P = P[mask]

    if valid_P.size > 0:
        # Calcul du coefficient binomial
        range_x = np.arange(1, x + 1)
        range_n = np.arange(n - x + 1, n + 1)
        log_comb = np.sum(np.log(range_n)) - np.sum(np.log(range_x))

        # Calcul uniquement pour les P valides
        log_likelihood = (
            log_comb
            + x * np.log(valid_P)
            + (n - x) * np.log(1 - valid_P)
        )
        results[mask] = np.exp(log_likelihood)

    # Gestion manuelle des cas limites
    results[P == 0] = 0.0 if x > 0 else 1.0
    results[P == 1] = 0.0 if x < n else 1.0

    return results
