#!/usr/bin/env python3
"""Détermine le meilleur nombre de clusters avec le BIC"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Calcule le BIC pour différents nombres de
    clusters et retourne le meilleur."""

    # Vérification des types et dimensions de X
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape

    # Vérification de kmin
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None, None, None

    # Initialisation de kmax si non fourni
    if kmax is None:
        kmax = n

    # Vérification de kmax
    if not isinstance(kmax, int) or kmax <= 0 or kmax < kmin or kmax > n:
        return None, None, None, None

    # Vérification que le nombre de valeurs de k est suffisant
    if kmax - kmin + 1 < 2:
        return None, None, None, None

    # Vérification des autres paramètres
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    # Initialisation des listes pour stocker les BIC et les log-vraisemblances
    b = []
    likelihoods = []

    # Boucle sur les différentes valeurs de k
    for k in range(kmin, kmax + 1):
        # Calcul de EM pour k clusters
        pi, m, S, g, li = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        # Vérification que l'EM a réussi
        if pi is None or m is None or S is None or g is None:
            return None, None, None, None

        # Calcul du nombre de paramètres pour le modèle
        p = (k * d) + (k * d * (d + 1) // 2) + (k - 1)

        # Calcul du BIC pour k clusters
        bic = p * np.log(n) - 2 * li

        # Stockage des log-vraisemblances et des BIC
        likelihoods.append(li)
        b.append(bic)

        # Sélection du meilleur modèle en fonction du BIC
        if k == kmin or bic < best_bic:
            best_bic = bic
            best_results = (pi, m, S)
            best_k = k

    # Conversion des listes en tableaux numpy
    likelihoods = np.array(likelihoods)
    b = np.array(b)

    # Retourne le meilleur k et les résultats associés
    return best_k, best_results, likelihoods, b
