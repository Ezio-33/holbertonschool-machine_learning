#!/usr/bin/env python3
"""Algorithme EM complet pour les Gaussian Mixture Models"""

import numpy as np
# Importation des fonctions d'initialisation, d'estimation et de maximisation
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization

def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Réalise l'algorithme d'Expectation-Maximization.

    Paramètres:
    X : np.ndarray
        Tableau de données de forme (n, d).
    k : int
        Nombre de mélanges (clusters).
    iterations : int, facultatif
        Nombre maximum d'itérations (défaut: 1000).
    tol : float, facultatif
        Tolérance pour la convergence de la log vraisemblance (défaut: 1e-5).
    verbose : bool, facultatif
        Si True, affiche la log vraisemblance périodiquement (défaut: False).

    Retourne:
      pi : np.ndarray
          Probabilités des mélanges.
      m : np.ndarray
          Moyennes des composantes.
      S : np.ndarray
          Matrices de covariance.
      g : np.ndarray
          Responsabilités (probabilités a posteriori).
      L : float
          Dernière log vraisemblance obtenue.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    prev_L = 0

    for i in range(1, iterations + 1):
        g, L = expectation(X, pi, m, S)

        # Affichage optionnel (chaque 10 itérations ou dernière)
        if verbose and (i % 10 == 0 or i == iterations):
            print(f"Log Likelihood after {i} iterations: {L:.5f}")

        if abs(L - prev_L) < tol:
            if verbose:
                print(f"Log Likelihood after {i} iterations: {L:.5f}")
            break

        prev_L = L

        pi, m, S = maximization(X, g)

    return pi, m, S, g, L