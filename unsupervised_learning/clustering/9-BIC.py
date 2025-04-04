#!/usr/bin/env python3
"""Détermine le meilleur nombre de clusters avec le BIC"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Calcule le BIC pour différentes valeurs de k et
    sélectionne la meilleure"""

    # Validation des entrées
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    n, d = X.shape

    if kmax is None:
        kmax = X.shape[0] - 1
    if kmin < 1 or kmax < kmin or kmax >= X.shape[0]:
        return None, None, None, None

    log_likelihoods = []
    bic_values = []
    results = []

    np.random.seed(11)  # Seed globale

    for k in range(kmin, kmax + 1):
        # Reset seed pour chaque k pour reproductibilité
        np.random.seed(11)

        # Calcul EM
        pi, m, S, _, l = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=False
        )

        if pi is None:
            continue

        # Calcul des paramètres
        params_pi = k - 1
        params_m = k * d
        params_S = k * d * (d + 1) // 2
        p = params_pi + params_m + params_S

        # Calcul BIC
        bic = p * np.log(n) - 2 * l

        # Stockage sans arrondi
        log_likelihoods.append(l)
        bic_values.append(bic)
        results.append((pi, m, S))

    if not bic_values:
        return None, None, None, None

    # Sélection du meilleur BIC
    best_idx = np.argmin(bic_values)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    # Conversion en arrays numpy
    log_likelihoods = np.array(log_likelihoods)
    bic_values = np.array(bic_values)

    return (best_k, best_result, log_likelihoods, bic_values)
