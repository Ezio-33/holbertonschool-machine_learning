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
        Tableau de données de forme (n, d) où n est le nombre
        d'exemples et d est la dimension.
    k : int
        Nombre de mélanges (clusters).
    iterations : int, facultatif
        Nombre maximum d'itérations. Par défaut à 1000.
    tol : float, facultatif
        Tolérance pour la convergence de la log vraisemblance.
        Par défaut à 1e-5.
    verbose : bool, facultatif
        Si True, affiche la log vraisemblance tous les 10 itérations.
        Par défaut à False.

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
        Dernière valeur de la log vraisemblance obtenue.
    """
    # Vérification des types et dimensions de X
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    # k doit être un entier positif
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    # iterations doit être un entier positif
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    # tol doit être un nombre positif ou nul
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    # verbose doit être un booléen
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # Initialisation des paramètres (pi, moyennes et covariances)
    pi, m, S = initialize(X, k)
    # Initialisation de la log vraisemblance précédente à 0
    prev_L = 0

    # Boucle principale pour l'algorithme EM
    for i in range(iterations):
        # Etape d'espérance (E-step) pour calculer les responsabilités et la
        # log vraisemblance
        g, L = expectation(X, pi, m, S)

        # Affichage optionnel de la log vraisemblance tous les 10 itérations ou
        # à la dernière itération
        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {L:.5f}")

        # Vérification de convergence : si la différence entre deux itérations
        # est inférieure à tol, on arrête
        if abs(L - prev_L) < tol:
            if verbose:
                print(f"Log Likelihood after {i} iterations: {L:.5f}")
            break

        # Mise à jour de la log vraisemblance précédente
        prev_L = L

        # Etape de maximisation (M-step) pour mettre à jour les paramètres
        pi, m, S = maximization(X, g)

    # Retourne les paramètres finaux, les responsabilités et la log
    # vraisemblance finale
    return pi, m, S, g, L
