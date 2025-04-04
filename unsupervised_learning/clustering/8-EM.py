#!/usr/bin/env python3
"""Algorithme EM complet pour les Gaussian Mixture Models"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Exécute l'algorithme EM complet pour un GMM

    Args:
        X (numpy.ndarray): Données de forme (n, d)
        k (int): Nombre de clusters
        iterations (int): Nombre maximal d'itérations
        tol (float): Seuil de convergence basé sur la log-vraisemblance
        verbose (bool): Affichage des informations de suivi

    Returns:
        tuple: (pi, m, S, g, l) ou None en cas d'erreur
    """
    # Validation des entrées
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return (None,) * 5
    if not isinstance(k, int) or k < 1:
        return (None,) * 5

    # Initialisation des paramètres
    pi, m, S = initialize(X, k)
    if pi is None:
        return (None,) * 5

    log_likelihood_prev = -np.inf
    log_likelihoods = []

    for i in range(iterations + 1):
        # Étape Expectation
        g, log_likelihood = expectation(X, pi, m, S)
        if g is None:
            return (None,) * 5

        # Vérification de la convergence
        if i > 0 and np.abs(log_likelihood - log_likelihood_prev) <= tol:
            if verbose and (i % 10 == 0 or i == iterations):
                print(
                    f"Log Likelihood after {i} iterations: {
                        log_likelihood:.5f}")
            break

        # Étape Maximization
        pi_new, m_new, S_new = maximization(X, g)
        if pi_new is None:
            return (None,) * 5

        # Mise à jour des paramètres
        pi, m, S = pi_new, m_new, S_new
        log_likelihood_prev = log_likelihood
        log_likelihoods.append(log_likelihood)

        # Affichage verbose
        if verbose and (i % 10 == 0 or i == iterations):
            print(f"Log Likelihood after {i} iterations: {log_likelihood:.5f}")

    return pi, m, S, g, log_likelihood
